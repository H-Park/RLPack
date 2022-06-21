cmake_minimum_required(VERSION 3.22)
project(RLPack)

# ----------------------------------- Includes for CMAKE ----------------------------------- #
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
# ----------------------------------- Set Global Variables for CMAKE ----------------------------------- #

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ----------------------------------- Package Dependencies ----------------------------------- #
find_package(Python REQUIRED COMPONENTS Interpreter NumPy Development)
find_package(PyBind11 REQUIRED)
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 61)
endif ()

find_package(Boost COMPONENTS program_options log log_setup mpi serialization REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DBOOST_LOG_DYN_LINK}")

# ----------------------------------- Add Modules and Link Libraries ----------------------------------- #
pybind11_add_module(
        RLPack

        src/binder.cpp

        src/DQN/Agent.cuh

        src/DQN/DQN1D/Dqn1D.cpp
        src/DQN/DQN1D/Dqn1D.h
)

add_dependencies(RLPack pybind11::headers)

target_link_libraries(RLPack PRIVATE Python::Python)
target_link_libraries(RLPack PRIVATE Python::NumPy)
target_link_libraries(RLPack PRIVATE "${OpenCV_LIBRARIES}")
target_link_libraries(RLPack PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(RLPack PRIVATE ${Boost_LOG_LIBRARY})
target_link_libraries(RLPack PRIVATE ${Boost_LOG_SETUP_LIBRARY})
target_link_libraries(RLPack PRIVATE ${Boost_LIBRARIES})
target_link_libraries(RLPack PRIVATE OpenMP::OpenMP_CXX)

# ----------------------------------- Installation ----------------------------------- #
install(
        TARGETS RLPack
        EXPORT RLPack
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)


if (CALL_FROM_SETUP_PY)
    set(RLPack_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
else ()
    set(PYBINDPLAYGROUND_INSTALL_PREFIX ${Python3_SITELIB})
endif ()

set_target_properties(
        RLPack PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION ON
        CXX_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN ON
)
