load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "nuplan_db_test_utils",
    srcs = ["nuplan_db_test_utils.py"],
    deps = [
        "//nuplan/database/maps_db:gpkg_mapsdb",
        "//nuplan/database/maps_db:imapsdb",
        "//nuplan/database/nuplan_db_orm:camera",
        "//nuplan/database/nuplan_db_orm:image",
        "//nuplan/database/nuplan_db_orm:lidar",
        "//nuplan/database/nuplan_db_orm:nuplandb",
        "//nuplan/database/nuplan_db_orm:nuplandb_wrapper",
    ],
)

py_test(
    name = "test_nuplan",
    size = "small",
    srcs = ["test_nuplan.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/database/nuplan_db_orm:nuplandb",
        "//nuplan/database/tests:nuplan_db_test_utils",
    ],
)
