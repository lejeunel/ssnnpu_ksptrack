#!/usr/bin/env python3

import pstats
p = pstats.Stats('log')
p.sort_stats('cumulative').print_stats(100)
