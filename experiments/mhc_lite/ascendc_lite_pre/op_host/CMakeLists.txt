if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        lite_pre_heads_def.cpp
    )
    add_ops_compile_options(
            OP_NAME LitePreHeads
            OPTIONS --cce-auto-sync=on
                    -Wno-deprecated-declarations
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_op_to_compiled_list()
    add_modules_sources(OPTYPE lite_pre_heads ACLNNTYPE aclnn)
endif()
