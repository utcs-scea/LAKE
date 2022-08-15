#include <linux/build-salt.h>
#include <linux/module.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

BUILD_SALT;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__attribute__((section(".gnu.linkonce.this_module"))) = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif

static const struct modversion_info ____versions[]
__used
__attribute__((section("__versions"))) = {
	{ 0x6c507e6a, "module_layout" },
	{ 0x8b59c3b2, "kmalloc_caches" },
	{ 0xd2b09ce5, "__kmalloc" },
	{ 0xd6ee688f, "vmalloc" },
	{ 0x15af343e, "param_ops_int" },
	{ 0x461d16ca, "sg_nents" },
	{ 0x5362b2f0, "crypto_unregister_template" },
	{ 0x867c87df, "filp_close" },
	{ 0x999e8297, "vfree" },
	{ 0x97651e6c, "vmemmap_base" },
	{ 0xd9a5ea54, "__init_waitqueue_head" },
	{ 0x6de13801, "wait_for_completion" },
	{ 0x13dd3a13, "param_ops_charp" },
	{ 0x7c32d0f0, "printk" },
	{ 0xf82848d8, "crypto_register_template" },
	{ 0x9369dd1d, "crypto_aead_setkey" },
	{ 0x7cd8d75e, "page_offset_base" },
	{ 0x12a38747, "usleep_range" },
	{ 0xdb7305a1, "__stack_chk_fail" },
	{ 0x2ea2c95c, "__x86_indirect_thunk_rax" },
	{ 0x98f44453, "crypto_destroy_tfm" },
	{ 0xbdfb6dbb, "__fentry__" },
	{ 0x40c7d84d, "kmem_cache_alloc_trace" },
	{ 0xcbd3efb, "crypto_alloc_aead" },
	{ 0x37a0cba, "kfree" },
	{ 0x7a4497db, "kzfree" },
	{ 0x29361773, "complete" },
	{ 0x342eb9cb, "aead_register_instance" },
	{ 0x5af8e1e9, "filp_open" },
};

static const char __module_depends[]
__used
__attribute__((section(".modinfo"))) =
"depends=";


MODULE_INFO(srcversion, "96E70EA9239A42BCC24982C");
