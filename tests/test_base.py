import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hawkes_tools.base import Base, ThreadPool, actual_kwargs


class BaseUtilitiesTest(unittest.TestCase):
    def test_base_helper_methods_match_standalone_surface(self):
        class Toy(Base):
            _attrinfos = {"name": {"writable": False}, "count": {}, "_private": {}}

            def __init__(self):
                self._set("name", self.__class__.__name__)
                self.count = 2
                self._private = "hidden"

        toy = Toy()

        self.assertEqual(toy.name, "Toy")
        self.assertRegex(toy._get_now(), r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{6}$")
        self.assertEqual(toy._as_dict(), {"name": "Toy", "count": 2})
        toy._inc_attr("count", 3)
        self.assertEqual(toy.count, 5)
        toy._set("name", "RenamedToy")
        self.assertEqual(toy.name, "RenamedToy")
        with self.assertRaisesRegex(ValueError, "key as string"):
            toy._set(3, "bad")

    def test_actual_kwargs_matches_reference_decorator_semantics(self):
        @actual_kwargs
        def f(arg1, arg2, kwarg1=None, kwarg2="", kwarg3=1, kwarg4=True):
            kwargs_ = sorted(f.actual_kwargs.items())
            all_kwargs_ = sorted(
                {
                    "arg1": arg1,
                    "arg2": arg2,
                    "kwarg1": kwarg1,
                    "kwarg2": kwarg2,
                    "kwarg3": kwarg3,
                    "kwarg4": kwarg4,
                }.items()
            )
            return kwargs_, all_kwargs_

        default_all_kwargs = [
            ("arg1", 1),
            ("arg2", 2),
            ("kwarg1", None),
            ("kwarg2", ""),
            ("kwarg3", 1),
            ("kwarg4", True),
        ]

        kwargs, all_kwargs = f(1, 2)
        self.assertEqual(kwargs, [])
        self.assertEqual(all_kwargs, default_all_kwargs)

        kwargs, all_kwargs = f(1, 2, kwarg2="value2", kwarg3=-3)
        self.assertEqual(kwargs, [("kwarg2", "value2"), ("kwarg3", -3)])
        expected = default_all_kwargs.copy()
        expected[3:5] = [("kwarg2", "value2"), ("kwarg3", -3)]
        self.assertEqual(all_kwargs, expected)

        with self.assertRaises(TypeError):
            f(1, 2, kwarg5="un_existing")

    def test_actual_kwargs_preserves_signature(self):
        @actual_kwargs
        def f(arg1, arg2, kwarg1=None):
            return arg1, arg2, kwarg1

        self.assertEqual(str(inspect.signature(f)), "(arg1, arg2, kwarg1=None)")

    def test_thread_pool_runs_work_with_optional_lock(self):
        pool = ThreadPool(with_lock=True, max_threads=3)
        values = []

        def append_value(value):
            with pool.lock:
                values.append(value)

        for value in range(10):
            pool.add_work(append_value, value)
        pool.start()

        self.assertEqual(sorted(values), list(range(10)))

    def test_thread_pool_validation_and_exception_propagation(self):
        with self.assertRaisesRegex(ValueError, "max_threads"):
            ThreadPool(max_threads=0)

        pool = ThreadPool(max_threads=1)
        with self.assertRaisesRegex(TypeError, "callable"):
            pool.add_work(3)

        pool.add_work(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        with self.assertRaisesRegex(RuntimeError, "boom"):
            pool.start()


if __name__ == "__main__":
    unittest.main()
