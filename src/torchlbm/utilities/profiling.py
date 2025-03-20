from torch.profiler import profile


def trace_handler(profiler: profile):
    output = profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)
    print(output)
    output = profiler.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_time_total", row_limit=2)
    print(output)
    # p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")
