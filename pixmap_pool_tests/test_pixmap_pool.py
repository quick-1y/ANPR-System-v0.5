import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtWidgets  # type: ignore

from anpr.ui.main_window import PixmapPool


class PixmapPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_does_not_exceed_limit_when_creating_new_pixmap(self) -> None:
        pool = PixmapPool(max_total_pixels=1_000)
        too_large = pool.acquire(QtCore.QSize(40, 40))  # 1600 пикселей
        self.assertIsNone(too_large)
        self.assertEqual(pool.total_pixels, 0)

    def test_reuses_and_counts_total_pixels(self) -> None:
        pool = PixmapPool(max_total_pixels=5_000)

        pixmap_one = pool.acquire(QtCore.QSize(40, 40))
        self.assertIsNotNone(pixmap_one)
        self.assertEqual(pool.total_pixels, 1_600)

        pool.release(pixmap_one)  # type: ignore[arg-type]
        pixmap_two = pool.acquire(QtCore.QSize(40, 40))
        self.assertIsNotNone(pixmap_two)
        self.assertEqual(pool.total_pixels, 1_600)

    def test_discards_cached_pixmaps_to_free_space(self) -> None:
        pool = PixmapPool(max_total_pixels=4_000)

        pixmap_small = pool.acquire(QtCore.QSize(50, 50))
        self.assertIsNotNone(pixmap_small)
        pool.release(pixmap_small)  # type: ignore[arg-type]
        self.assertGreater(pool.total_pixels, 0)

        pixmap_large = pool.acquire(QtCore.QSize(60, 60))
        self.assertIsNotNone(pixmap_large)
        self.assertLessEqual(pool.total_pixels, 4_000)

        self.assertNotIn((50, 50), pool._pool)


if __name__ == "__main__":
    unittest.main()
