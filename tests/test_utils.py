import numpy as np

from shapely.geometry import box

from wake_uncertainty.utils import place_turbines, make_neighbour_polygon


def test_place_exact_number_of_turbines():
    polygon = box(0, 0, 10000, 10000)

    x, y, _ = place_turbines(polygon=polygon, n_turbines=44, min_spacing_m=500, seed=42)

    assert len(x) == 44
    assert len(y) == 44


def test_place_turbines_reproducible():
    polygon = box(0, 0, 10000, 10000)

    x1, y1, _ = place_turbines(polygon, 20, 500, seed=42)
    x2, y2, _ = place_turbines(polygon, 20, 500, seed=42)

    assert np.allclose(x1, x2)
    assert np.allclose(y1, y2)


def test_different_seed_changes_layout():
    polygon = box(0, 0, 10000, 10000)

    x1, y1, _ = place_turbines(polygon, 20, 500, seed=42)
    x2, y2, _ = place_turbines(polygon, 20, 500, seed=43)

    assert not (np.allclose(x1, x2) and np.allclose(y1, y2))


def test_minimum_spacing_is_respected():
    polygon = box(0, 0, 20000, 20000)
    spacing = 800

    x, y, _ = place_turbines(polygon, 30, spacing, seed=42)

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            distance = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            assert distance >= spacing


def test_turbines_inside_polygon():
    from shapely.geometry import Point

    polygon = box(0, 0, 10000, 10000)
    x, y, returned_polygon = (place_turbines(polygon, 20, 500, seed=42))

    for tx, ty in zip(x, y):
        assert returned_polygon.contains(Point(tx, ty))


def test_neighbour_polygon_has_positive_area():
    polygon, cx, cy = make_neighbour_polygon(0, 0, distance_km=15,
                                             direction_deg=270, n_turbines=60,spacing_D=7,
                                             rotor_dia=178)

    assert polygon.is_valid
    assert polygon.area > 0


def test_neighbour_distance():
    polygon, cx, cy = make_neighbour_polygon(0, 0, distance_km=15,
                                             direction_deg=270, n_turbines=60, spacing_D=7,
                                             rotor_dia=178)

    distance = np.sqrt(cx ** 2 + cy ** 2)
    assert np.isclose(distance, 15000)
