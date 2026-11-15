use crate::{cstr, sys};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Gpu,
    IntegratedGpu,
    Accelerator,
    Other,
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub index: usize,
    /// ggml's device name, e.g. "CPU", "CUDA0", "Vulkan0", "Metal".
    pub name: String,
    pub description: String,
    pub kind: DeviceKind,
    pub memory_total: u64,
    pub memory_free: u64,
}

/// Every device in the ggml backends compiled into this binary.
pub fn list_devices() -> Vec<DeviceInfo> {
    let n = unsafe { sys::ggml_backend_dev_count() };
    (0..n)
        .map(|i| unsafe {
            let dev = sys::ggml_backend_dev_get(i);
            let mut p: sys::ggml_backend_dev_props = std::mem::zeroed();
            sys::ggml_backend_dev_get_props(dev, &mut p);
            let kind = match p.type_ {
                sys::ggml_backend_dev_type::GGML_BACKEND_DEVICE_TYPE_CPU => DeviceKind::Cpu,
                sys::ggml_backend_dev_type::GGML_BACKEND_DEVICE_TYPE_GPU => DeviceKind::Gpu,
                sys::ggml_backend_dev_type::GGML_BACKEND_DEVICE_TYPE_IGPU => DeviceKind::IntegratedGpu,
                sys::ggml_backend_dev_type::GGML_BACKEND_DEVICE_TYPE_ACCEL => DeviceKind::Accelerator,
                _ => DeviceKind::Other,
            };
            DeviceInfo {
                index: i,
                name: cstr(p.name),
                description: cstr(p.description),
                kind,
                memory_total: p.memory_total as u64,
                memory_free: p.memory_free as u64,
            }
        })
        .collect()
}
