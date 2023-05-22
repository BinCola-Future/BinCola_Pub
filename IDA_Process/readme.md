# IDA scripts

## Environmental preparation

- capstone
- networkx

## Basic function features

```python
1. Presematic_features_extract()
    {'bin_path':bin_path,
     'bfunc_data_dict':{func_name:bfunc_data}}
    bfunc_data: class BFunc()
    feature/
        1. CFG-13
            # num of basic blocks, edges, loops, SCCs, and back edges
            - cfg_size
            - cfg_num_loops
            - cfg_num_loops_inter
            - cfg_num_scc
            - cfg_num_backedges
            # Avg./num of edges per a basic block
            - cfg_avg_degree
            - cfg_num_degree
            # Avg./Sum of basic block, loop, and SCC sizes
            - cfg_avg_loopintersize
            - cfg_avg_loopsize
            - cfg_avg_sccsize
            - cfg_sum_loopintersize
            - cfg_sum_loopsize
            - cfg_sum_sccsize
        2. Instrs-28
            # num of all, arith, data transfer, cmp, and logic instrs.
            - inst_num_inst
            - inst_num_arith
            - inst_num_dtransfer
            - inst_num_cmp
            - inst_num_logic
            # num of shift, bit-manipulating, flfloat, misc instrs.
            - inst_num_shift
            - inst_num_bitflag
            - inst_num_floatinst
            - inst_num_misc
            # num of arith + shift, and data transfer + misc instrs.
            - inst_num_abs_arith
            - inst_num_abs_dtransfer
            # num of all/unconditional/conditional control transfer instrs.
            - inst_num_ctransfer
            - inst_num_cndctransfer
            # Avg./num of all, arith, data transfer, cmp, and logic instrs. 
            - inst_avg_inst
            - inst_avg_arith
            - inst_avg_floatinst
            - inst_avg_dtransfer
            - inst_avg_cmp
            - inst_avg_logic
            # Avg./num of shift, bit-manipulating, flfloat, misc instrs. 
            - inst_avg_shift
            - inst_avg_bitflag
            - inst_avg_misc
            # Avg./num of arith + shift, and data transfer + misc instrs. 
            - inst_avg_abs_arith
            - inst_avg_abs_dtransfer
            # Avg./num of all/unconditional/conditional control transfer instrs. 
            - inst_avg_abs_ctransfer
            - inst_num_abs_ctransfer
            - inst_avg_cndctransfer
            - inst_avg_ctransfer
        3. CG-6
            # num of callers, callees, imported callees
            - cg_num_callers
            - cg_num_callees
            - cg_num_imported_callees
            # num of incoming/outgoing/imported calls
            - cg_num_incalls
            - cg_num_outcalls
            - cg_num_imported_calls
