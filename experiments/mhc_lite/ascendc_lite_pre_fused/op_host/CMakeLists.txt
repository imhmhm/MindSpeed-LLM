if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        lite_pre_fused_def.cpp
    )
    add_ops_compile_options(
            OP_NAME LitePreFused
            OPTIONS --cce-auto-sync=on
                    -Wno-deprecated-declarations
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_op_to_compiled_list()
    add_modules_sources(OPTYPE lite_pre_fused ACLNNTYPE aclnn)
endif()
