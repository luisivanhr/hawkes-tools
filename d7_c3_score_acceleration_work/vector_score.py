"""Shared-trace complete 25-coordinate score with a batched exact recursion."""
from __future__ import annotations
import math
from typing import Any, Mapping, Sequence
import numpy as np
from directional_score_reference import _weight_stats
from exact_trace_density import trace_log_density


def _state_quantities(evaluator: Any, ages: np.ndarray, time: float, event_history: np.ndarray, *, hazard: bool = False):
    method = evaluator.log_hazard_and_gradient if hazard else evaluator.cumulative_hazard_and_gradient
    first, first_gradient = method(ages + float(time))
    count = event_history.size + 1
    value = np.empty((ages.size, count), dtype=float)
    gradient = np.empty((ages.size, count, evaluator.dimension), dtype=float)
    value[:, 0], gradient[:, 0] = first, first_gradient
    if event_history.size:
        common, common_gradient = method(float(time) - event_history)
        value[:, 1:] = np.asarray(common)[None, :]
        gradient[:, 1:] = np.asarray(common_gradient)[None, :, :]
    return value, gradient


def prehistory_statistics(traces: Sequence[Any], kernel: Any):
    rates = np.asarray(kernel.component_rates, dtype=float)
    zero = np.zeros((len(traces), 2)); first = np.zeros_like(zero)
    for i, trace in enumerate(traces):
        pre = np.asarray(trace.cluster_event_times, dtype=float)
        if pre.size:
            ex = np.exp(pre[:, None] * rates[None, :])
            zero[i] = np.sum(ex, axis=0)
            first[i] = np.sum(pre[:, None] * ex, axis=0)
    return zero, first


def kernel_window_batch(times: np.ndarray, horizon: float, kernel: Any, zero: np.ndarray, first: np.ndarray):
    sample_count = zero.shape[0]
    masses = np.asarray(kernel.component_masses, dtype=float)
    rates = np.asarray(kernel.component_rates, dtype=float)
    mass_derivative = np.asarray([[kernel.slow_weight, kernel.alpha, 0., 0.], [1.-kernel.slow_weight, -kernel.alpha, 0., 0.]])
    intensity = np.zeros((sample_count, times.size))
    derivative = np.zeros((sample_count, times.size, 4))
    compensator = np.zeros(sample_count); comp_gradient = np.zeros((sample_count, 4))
    for k, (mass, rate) in enumerate(zip(masses, rates, strict=True)):
        for j, time in enumerate(times):
            decay = math.exp(-rate * float(time))
            unit = rate * decay * zero[:, k]
            rate_gradient = decay * ((1.-rate*time) * zero[:, k] + rate*first[:, k])
            if j:
                lag = time-times[:j]; ex=np.exp(-rate*lag)
                unit=unit+rate*float(np.sum(ex))
                rate_gradient=rate_gradient+float(np.sum(ex*(1.-rate*lag)))
            intensity[:, j] += mass*unit
            derivative[:, j] += unit[:, None]*mass_derivative[k]
            derivative[:, j, 2+k] += mass*rate_gradient
        decay=math.exp(-rate*horizon)
        unit=zero[:, k]*(1.-decay)
        rate_gradient=first[:, k]*(1.-decay)+horizon*decay*zero[:, k]
        if times.size:
            remaining=horizon-times; ex=np.exp(-rate*remaining)
            unit=unit+float(np.sum(1.-ex))
            rate_gradient=rate_gradient+float(np.sum(remaining*ex))
        compensator += mass*unit
        comp_gradient += unit[:, None]*mass_derivative[k]
        comp_gradient[:, 2+k] += mass*rate_gradient
    return intensity, derivative, compensator, comp_gradient


def conditional_batch(times: np.ndarray, horizon: float, traces: Sequence[Any], evaluator: Any, kernel: Any, prehistory=None):
    times=np.asarray(times,dtype=float); horizon=float(horizon)
    ages=np.asarray([trace.age for trace in traces],dtype=float)
    if times.ndim!=1 or np.any(times<=0) or np.any(times>=horizon) or np.any(np.diff(times)<=0):
        raise ValueError("invalid observed-window chronology")
    zero,first=prehistory if prehistory is not None else prehistory_statistics(traces,kernel)
    offspring, offspring_gradient, compensator, comp_gradient=kernel_window_batch(times,horizon,kernel,zero,first)
    t=ages.size; renewal_dimension=evaluator.dimension; dimension=renewal_dimension+4
    weights=np.ones((t,1)); weight_gradient=np.zeros((t,1,dimension))
    log_value=np.zeros(t); log_gradient=np.zeros((t,dimension)); previous=0.
    for j,current in enumerate(times):
        history=times[:j]
        old,old_gradient=_state_quantities(evaluator,ages,previous,history)
        new,new_gradient=_state_quantities(evaluator,ages,float(current),history)
        delta_gradient=np.zeros(weight_gradient.shape)
        delta_gradient[:,:,:renewal_dimension]=new_gradient-old_gradient
        survived=np.exp(-(new-old))
        pre_weights=weights*survived
        pre_gradient=survived[:,:,None]*(weight_gradient-weights[:,:,None]*delta_gradient)
        log_hazard, log_hazard_gradient=_state_quantities(evaluator,ages,float(current),history,hazard=True)
        hazard=np.exp(log_hazard)
        hazard_gradient=np.zeros(weight_gradient.shape)
        hazard_gradient[:,:,:renewal_dimension]=hazard[:,:,None]*log_hazard_gradient
        off=offspring[:,j]
        off_gradient=np.zeros((t,dimension)); off_gradient[:,renewal_dimension:]=offspring_gradient[:,j]
        rate=hazard+off[:,None]
        mass=np.sum(pre_weights*rate,axis=1)
        if np.any(~np.isfinite(mass)) or np.any(mass<=0): raise FloatingPointError("nonpositive event mass")
        gradient=np.sum(pre_gradient*rate[:,:,None]+pre_weights[:,:,None]*(hazard_gradient+off_gradient[:,None,:]),axis=1)
        score=gradient/mass[:,None]
        log_value+=np.log(mass); log_gradient+=score
        old_numerator=pre_weights*off[:,None]
        old_numerator_gradient=pre_gradient*off[:,None,None]+pre_weights[:,:,None]*off_gradient[:,None,:]
        old_weights=old_numerator/mass[:,None]
        old_weight_gradient=old_numerator_gradient/mass[:,None,None]-old_weights[:,:,None]*score[:,None,:]
        new_numerator=np.sum(pre_weights*hazard,axis=1)
        new_numerator_gradient=np.sum(pre_gradient*hazard[:,:,None]+pre_weights[:,:,None]*hazard_gradient,axis=1)
        new_weight=new_numerator/mass
        new_weight_gradient=new_numerator_gradient/mass[:,None]-new_weight[:,None]*score
        weights=np.concatenate((old_weights,new_weight[:,None]),axis=1)
        weight_gradient=np.concatenate((old_weight_gradient,new_weight_gradient[:,None,:]),axis=1)
        previous=float(current)
    old,old_gradient=_state_quantities(evaluator,ages,previous,times)
    final,final_gradient=_state_quantities(evaluator,ages,horizon,times)
    delta_gradient=np.zeros(weight_gradient.shape); delta_gradient[:,:,:renewal_dimension]=final_gradient-old_gradient
    survival=np.exp(-(final-old)); mass=np.sum(weights*survival,axis=1)
    if np.any(mass<=0): raise FloatingPointError("nonpositive terminal mass")
    gradient=np.sum(survival[:,:,None]*(weight_gradient-weights[:,:,None]*delta_gradient),axis=1)
    log_value+=np.log(mass)-compensator
    log_gradient+=gradient/mass[:,None]
    log_gradient[:,renewal_dimension:]-=comp_gradient
    return log_value,log_gradient


def evaluate_archive(point: np.ndarray, *, context: Any, archive: Any, windows: Sequence[Mapping[str,Any]], adapter: Any, clipping_cap: float):
    score_adapter=adapter.score_adapter; analytic=score_adapter.analytic; core=adapter.d7
    law,kernel,_=core.model_from_normalized(point,context)
    evaluator=analytic.RenewalScoreEvaluator(law)
    jacobian=score_adapter.normalized_decode_jacobian(point,context)
    trace_log=np.asarray([trace_log_density(trace,law,kernel,core.base) for trace in archive.traces])
    proposal=np.asarray(archive.proposal_log_density)
    trace_gradients=np.stack([analytic.boundary_trace_score(trace,evaluator,kernel) for trace in archive.traces])@jacobian
    conditional=np.empty((len(windows),archive.size)); gradients=np.empty(conditional.shape+(25,))
    stats=prehistory_statistics(archive.traces,kernel)
    excitation=core.base.boundary_excitation_batch(archive.traces,kernel)
    parity=0.; gradient_parity=0.; clipped=0
    for w,window in enumerate(windows):
        values,physical=conditional_batch(window["events"],float(window["horizon"]),archive.traces,evaluator,kernel,stats)
        reference=core.base.conditional_log_density_batch(archive.traces,window,law,kernel,excitation=excitation)
        parity=max(parity,float(np.max(np.abs(values-reference))))
        conditional[w]=values; gradients[w]=physical@jacobian
        if w in (0,len(windows)-1):
            value,gradient=analytic.conditional_log_density_and_gradient(window["events"],float(window["horizon"]),archive.traces[0],evaluator,kernel)
            parity=max(parity,abs(float(values[0])-float(value)))
            gradient_parity=max(gradient_parity,float(np.max(np.abs(gradients[w,0]-np.asarray(gradient)@jacobian))))
        base_reference=core.base._window_reference_log_density(window,law,kernel)
        clipped+=int(np.count_nonzero(values+trace_log-proposal-base_reference>math.log(clipping_cap)))
    arrays={"log_weights":conditional+trace_log[None,:]-proposal[None,:],"complete_gradients":gradients+trace_gradients[None,:,:],"trace_gradients":trace_gradients,"conditional_gradients":gradients,"conditional_log_likelihood":conditional,"source_trace_log_density":trace_log,"proposal_log_density":proposal}
    if any(np.any(~np.isfinite(row)) for row in arrays.values()): raise FloatingPointError("nonfinite complete-gradient archive")
    diagnostics={"archive_size":archive.size,"archive_seed":archive.seed,"trace_archive_sha256":score_adapter.trace_archive_sha256(archive),"proposal_identity_error":float(np.max(np.abs(trace_log-proposal))),"conditional_value_parity_error":parity,"conditional_gradient_parity_error":gradient_parity,"clipped_count":clipped,"full_boundary_and_conditional_terms":True}
    return arrays,diagnostics


def _vector_ratio(logw: np.ndarray, complete: np.ndarray, trace: np.ndarray, indices: np.ndarray):
    w_count,n,dimension=complete.shape
    raw=np.empty((w_count,dimension)); cv=np.empty_like(raw); bias=np.empty_like(raw)
    window_cov=np.empty((w_count,dimension,dimension)); aggregate_if=np.zeros((n,dimension)); path_if=np.zeros((16,n,dimension))
    trace_mean=np.mean(trace,axis=0); trace_total=np.sum(trace,axis=0)
    statistics=[]; logz=[]
    for w in range(w_count):
        maximum=float(np.max(logw[w])); weight=np.exp(logw[w]-maximum); denominator=float(np.sum(weight))
        numerator=np.sum(weight[:,None]*complete[w],axis=0)
        raw[w]=numerator/denominator; cv[w]=raw[w]-trace_mean
        influence=(weight/(denominator/n))[:,None]*(complete[w]-raw[w])-trace
        window_cov[w]=np.cov(influence,rowvar=False,ddof=1)/n
        aggregate_if+=influence/w_count
        path_if[int(indices[w])]+=influence/np.count_nonzero(indices==indices[w])
        loo=(numerator[None,:]-weight[:,None]*complete[w])/(denominator-weight)[:,None]-(trace_total[None,:]-trace)/(n-1)
        bias[w]=(n-1)*(np.mean(loo,axis=0)-cv[w])
        statistics.append(_weight_stats(logw[w])); logz.append(maximum+math.log(denominator/n))
    paths=np.stack([np.mean(cv[indices==i],axis=0) for i in range(16)])
    return {"gradient":np.mean(cv,axis=0),"raw_gradient":np.mean(raw,axis=0),"gradient_covariance":np.cov(aggregate_if,rowvar=False,ddof=1)/n,"gradient_bias":np.mean(bias,axis=0),"window_gradients":cv,"window_covariances":window_cov,"window_bias":bias,"path_gradients":paths,"path_covariances":np.stack([np.cov(row,rowvar=False,ddof=1)/n for row in path_if]),"trace_mean":trace_mean,"trace_covariance":np.cov(trace,rowvar=False,ddof=1)/n,"weight_stats":statistics,"log_normalizers":np.asarray(logz),"sample_count":n}


def summarize(replicate_chunks: Sequence[Sequence[Mapping[str,np.ndarray]]], indices: np.ndarray):
    raw_replicates=[]
    for chunks in replicate_chunks:
        raw_replicates.append({"logw":np.concatenate([x["log_weights"] for x in chunks],axis=1),"complete":np.concatenate([x["complete_gradients"] for x in chunks],axis=1),"trace":np.concatenate([x["trace_gradients"] for x in chunks],axis=0)})
    replicas=[_vector_ratio(x["logw"],x["complete"],x["trace"],indices) for x in raw_replicates]
    pooled=_vector_ratio(np.concatenate([x["logw"] for x in raw_replicates],axis=1),np.concatenate([x["complete"] for x in raw_replicates],axis=1),np.concatenate([x["trace"] for x in raw_replicates],axis=0),indices)
    arrays={key:np.asarray(pooled[key]) for key in ("gradient","raw_gradient","gradient_covariance","gradient_bias","window_gradients","window_covariances","window_bias","path_gradients","path_covariances","trace_mean","trace_covariance","log_normalizers")}
    arrays["replicate_gradients"]=np.stack([x["gradient"] for x in replicas])
    stats=pooled["weight_stats"]; rs=[r for x in replicas for r in x["weight_stats"]]
    summary={"sample_count_total":pooled["sample_count"],"sample_count_per_replicate":replicas[0]["sample_count"],"gradient":arrays["gradient"],"gradient_norm":float(np.linalg.norm(arrays["gradient"])),"maximum_coordinate_se":float(np.max(np.sqrt(np.diag(arrays["gradient_covariance"])))),"minimum_pooled_absolute_ess":min(x["absolute_ess"] for x in stats),"minimum_replicate_absolute_ess":min(x["absolute_ess"] for x in rs),"maximum_pooled_single_weight_share":max(x["maximum_weight_share"] for x in stats),"maximum_replicate_single_weight_share":max(x["maximum_weight_share"] for x in rs),"maximum_pooled_top_point_one_percent_share":max(x["top_point_one_percent_share"] for x in stats),"maximum_pooled_top_one_percent_share":max(x["top_one_percent_share"] for x in stats),"maximum_replicate_top_point_one_percent_share":max(x["top_point_one_percent_share"] for x in rs),"maximum_replicate_top_one_percent_share":max(x["top_one_percent_share"] for x in rs),"all_finite":all(np.all(np.isfinite(x)) for x in arrays.values())}
    return summary,arrays


def directional(arrays: Mapping[str,np.ndarray], direction: np.ndarray, previous_gradient=None):
    d=np.asarray(direction,dtype=float)
    score=float(arrays["gradient"]@d); se=math.sqrt(max(0.,float(d@arrays["gradient_covariance"]@d)))
    bias=float(arrays["gradient_bias"]@d)
    change=0. if previous_gradient is None else abs(float((arrays["gradient"]-previous_gradient)@d))
    paths=arrays["path_gradients"]@d; folds=np.asarray([np.mean(paths[f::4]) for f in range(4)])
    margin=2.131449545559323*float(np.std(paths,ddof=1)/4.)
    replicates=arrays["replicate_gradients"]@d
    allowance=4.*se+abs(bias)+change
    trace=float(arrays["trace_mean"]@d); trace_se=math.sqrt(max(0.,float(d@arrays["trace_covariance"]@d)))
    return {"score":score,"numerical_standard_error":se,"jackknife_bias":bias,"absolute_last_level_change":change,"numerical_allowance":allowance,"numerical_interval":[score-allowance,score+allowance],"across_path_interval":[score-margin,score+margin],"path_scores":paths,"fold_means":folds,"replicate_scores":replicates,"replicate_range":float(np.ptp(replicates)),"maximum_window_se":float(np.max(np.sqrt(np.maximum(0.,np.einsum('i,wij,j->w',d,arrays["window_covariances"],d))))),"maximum_window_bias":float(np.max(np.abs(arrays["window_bias"]@d))),"trace_martingale_standardized":abs(trace)/max(trace_se,1e-300)}
