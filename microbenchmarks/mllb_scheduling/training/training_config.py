fields=['src_non_pref_nr', 'src_non_numa_nr', 'delta_hot', 'cpu_idle',
          'cpu_not_idle', 'cpu_newly_idle', 'same_node', 'prefer_src', 'prefer_dst',
          'p_running',
          'imbalance', 'total_faults',
          'src_len',
          'src_load', 'dst_load',
          'throttled',
          'dst_len',
          'test_aggressive',
          'delta_faults', 'extra_fails', 'buddy_hot']
label=['can_migrate']

features=[
    'src_non_pref_nr',
    #  'src_non_numa_nr',
    'delta_hot',
    'cpu_idle',
    'cpu_not_idle',
    'cpu_newly_idle',
    'same_node', 'prefer_src', 'prefer_dst',
    #  'imbalance',
    #  'total_faults',
    'src_len',
    'src_load',
    'dst_load',
    #  'throttled',
    'dst_len',
    'delta_faults', 'extra_fails', 'buddy_hot'
]

columns = features + label

#final
"""
features=[
    'src_non_pref_nr',
    'delta_hot',
    'cpu_idle',
    'cpu_not_idle',
    'cpu_newly_idle',
    'same_node', 'prefer_src', 'prefer_dst',
    'src_len',
    'src_load',
    'dst_load',
    'dst_len',
    'delta_faults', 'extra_fails', 'buddy_hot'
]
"""
