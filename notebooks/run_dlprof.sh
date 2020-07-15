nsys profile -t osrt,cuda,nvtx -c cudaProfilerApi --stop-on-range-end true --show-output=true --export=sqlite -o ./nsys_profile python punctuation_pyprof_o0.py
dlprof --nsys_database=nsys_profile.sqlite
mv event_files event_files_o0
mv nsys_profile.qdrep nsys_profile_o0.qdrep
mv nsys_profile.sqlite nsys_profile_o0.sqlite


nsys profile -t osrt,cuda,nvtx -c cudaProfilerApi --stop-on-range-end true --show-output=true --export=sqlite -o ./nsys_profile python punctuation_pyprof_o1.py
dlprof --nsys_database=nsys_profile.sqlite
mv event_files event_files_o1
mv nsys_profile.qdrep nsys_profile_o1.qdrep
mv nsys_profile.sqlite nsys_profile_o1.sqlite
