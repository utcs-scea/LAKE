/*
 * Part of LAKE: Towards a Machine Learning-Assisted Kernel with LAKE
 * Copyright (C) 2022-2024 Henrique Fingler
 * Copyright (C) 2022-2024 Isha Tarte
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */


#include <linux/module.h>
#include <linux/delay.h>
#include "lake_kapi.h"
#include "kargs.h"
#include "cuda.h"

static int __init lake_kapi_init(void)
{
    int err;
	err = lake_init_socket();
    init_kargs_kv();
    if (err < 0) {
		printk(KERN_ERR "Err in init_socket %d\n", err);
        return -1;
	}

    pr_info("[lake] Registered CUDA kapi\n");
    
    return 0;
}

static void __exit lake_kapi_exit(void)
{
    destroy_kargs_kv();
	lake_destroy_socket();
}

MODULE_AUTHOR("Henrique Fingler");
MODULE_DESCRIPTION("LAKE CUDA kapi");
MODULE_LICENSE("GPL");
module_init(lake_kapi_init)
module_exit(lake_kapi_exit)
