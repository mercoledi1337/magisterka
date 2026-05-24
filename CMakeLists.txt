cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(example-app)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example-app example-app.cpp
        main.cpp
        main.h
        core/Model.h
        core/Trainer.h
        core/Trainer.cpp
        core/Evaluator.cpp
        core/Evaluator.h
        dataset/DataLoader.cpp
        dataset/DataLoader.h
        dataset/Batch.h
        cnn/ConvLayer.h
        cnn/ConvLayer.cpp
        cnn/PoolLayer.h
        cnn/PollLayer.cpp
        cnn/CNN.h
        cnn/CNN.cpp
        resnet/ResidualBlock.h
        resnet/ResNet.h
        resnet/ResNet.cpp
        mobilenet/DepthwiseConv.h
        mobilenet/MobileNet.h
        mobilenet/MobileNet.cpp
        efficientnet/EfficientNet.h
        efficientnet/EfficientNet.cpp
        utils/ImageUtils.h
        utils/ImageUtils.h.cpp
        utils/Metrics.h
        utils/Metrics.cpp
        core/Loss.h
        core/Loss.cpp
        cnn/Flatten.h
        cnn/Flatten.cpp
        resnet/ResidualBlock.cpp
        mobilenet/DepthwiseConv.cpp
        mobilenet/PointwiseConv.h
        mobilenet/PointwiseConv.cpp
        efficientnet/MBConv.h
        efficientnet/MBConv.cpp
        efficientnet/SEBlock.h
        efficientnet/SEBlock.cpp
        dataset/Dataset.h
        dataset/Dataset.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    # Znajdź ścieżkę do folderu z bibliotekami (tam gdzie są DLL)
    # find_package(Torch) ustawia zmienną TORCH_INSTALL_PREFIX
    set(TORCH_DLL_DIR "${TORCH_INSTALL_PREFIX}/lib")

    file(GLOB TORCH_DLLS "${TORCH_DLL_DIR}/*.dll")

    if (TORCH_DLLS)
        add_custom_command(TARGET example-app
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS}
                $<TARGET_FILE_DIR:example-app>)
    endif()
endif (MSVC)