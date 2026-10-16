/*
 * Apple AGP bridge stub: a Uni-North AGP host bridge as a plain PCI device
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * Why it exists: MorphOS 3.20's Radeon driver sets up its DMA command ring
 * only after pcix.library's FindBoardTagList() returns an Apple AGP bridge
 * board (vendor 0x106B, device 0x002D/0x0034/0x0066 or 0x004B/0x0059).
 * MorphOS builds its board list from the OpenBIOS device tree, and OpenBIOS
 * never gives a host bridge a node of its own, so the real uni-north-agp
 * bridge can never be found whatever its ID.  This stub sits on the scanned
 * PCI bus as an ordinary device carrying the bridge's ID, AGP capability and
 * GART registers; its GART writes go through the real bridge's handler into
 * uninorth_get_agp_gart_base(), which the GPU model's AGP translation reads.
 *
 * Class code: OpenBIOS takes any class 0x0600 device it meets on a bus for
 * the host bridge seen again and drops it ("host bridge already
 * configured"), so the default is 0x0680 (other bridge); class-id=0x0600
 * gives the real chip's class.
 *
 *   -device apple-agp-bridge,addr=10.0      (mac99 main PCI bus)
 */

#include "qemu/osdep.h"
#include "hw/pci/pci_device.h"
#include "hw/pci/pci_ids.h"
#include "hw/pci-host/uninorth.h"
#include "hw/qdev-properties.h"
#include "qemu/module.h"

typedef struct AppleAGPBridge {
    PCIDevice parent_obj;
    uint16_t device_id;
    uint16_t class_id;
} AppleAGPBridge;

static void apple_agp_bridge_realize(PCIDevice *d, Error **errp)
{
    AppleAGPBridge *s = (AppleAGPBridge *)d;

    pci_config_set_device_id(d->config, s->device_id);
    pci_config_set_class(d->config, s->class_id);
    unin_add_agp_capability(d, errp);
}

static const Property apple_agp_bridge_props[] = {
    /* 0x0034 = UniNorth 2: "flavour 0", little-endian `phys | 1` entries */
    DEFINE_PROP_UINT16("device-id", AppleAGPBridge, device_id, 0x0034),
    DEFINE_PROP_UINT16("class-id", AppleAGPBridge, class_id,
                       PCI_CLASS_BRIDGE_OTHER),
};

static void apple_agp_bridge_class_init(ObjectClass *klass, void *data)
{
    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);
    DeviceClass *dc = DEVICE_CLASS(klass);

    k->realize = apple_agp_bridge_realize;
    k->config_write = unin_agp_pci_host_config_write;
    k->vendor_id = PCI_VENDOR_ID_APPLE;
    /* revision 0, no BARs, no interrupt pin */
    device_class_set_props(dc, apple_agp_bridge_props);
    set_bit(DEVICE_CATEGORY_BRIDGE, dc->categories);
}

static const TypeInfo apple_agp_bridge_info = {
    .name = "apple-agp-bridge",
    .parent = TYPE_PCI_DEVICE,
    .instance_size = sizeof(AppleAGPBridge),
    .class_init = apple_agp_bridge_class_init,
    .interfaces = (InterfaceInfo[]) {
        { INTERFACE_CONVENTIONAL_PCI_DEVICE },
        { },
    },
};

static void apple_agp_bridge_register_types(void)
{
    type_register_static(&apple_agp_bridge_info);
}

type_init(apple_agp_bridge_register_types)
