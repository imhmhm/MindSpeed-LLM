add_ops_compile_options(
        OP_NAME LitePreGrad
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
)

target_sources(op_host_aclnn PRIVATE
op_host/lite_pre_grad.cpp
)

target_sources(optiling PRIVATE
        op_host/lite_pre_grad.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_sources(opsproto PRIVATE
         op_host/lite_pre_grad.cpp
)

install(FILES op_kernel/lite_pre_grad.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
