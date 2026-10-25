add_library(RopeOnTableCaseStudy STATIC RopeOnTableCaseStudy.cpp)
target_sources(RopeOnTableCaseStudy INTERFACE RopeOnTableCaseStudy.hpp)

target_link_libraries(RopeOnTableCaseStudy
        PRIVATE
        Utilities
        Simulator
        Systems
        Materials
        ElasticaFlags
        )
# So that includes can be written as <elastica/CaseStudy.hpp>
target_include_directories(RopeOnTableCaseStudy PRIVATE ${CMAKE_SOURCE_DIR})

include(ElasticaCaseStudyHelpers)

add_elastica_case_study(example_rope_on_table
        FROM RopeOnTableCaseStudy
        AS elastica::standalone_executable
        )

add_elastica_case_study(PyRopeOnTableCaseStudy
        FROM RopeOnTableCaseStudy
        AS elastica::python_object
        )

elastica_python_link_libraries(PyRopeOnTableCaseStudy PRIVATE Blaze)
elastica_python_add_dependencies(PyRopeOnTableCaseStudy PyArrays)
