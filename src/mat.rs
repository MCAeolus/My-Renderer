use crate::structs::{Mat4, Vec3, Vec4, Transform};

pub fn homogeneous_scale(scale: f32) -> Mat4 {
    Mat4::new_scaling(scale)
}

pub fn get_transform_mat(transform: &Transform) -> Mat4 {
    let mut mat: nalgebra::Matrix<f32, nalgebra::Const<4>, nalgebra::Const<4>, nalgebra::ArrayStorage<f32, 4, 4>> = Mat4::new_translation(&transform.position);
    mat *= transform.rotation.to_homogeneous();
    mat *= Mat4::new_scaling(transform.scale);
    mat
}

#[inline]
pub fn canonicalize(vec: &Vec4) -> Vec4 {
    vec.scale(1./vec.w)
}

#[inline]
pub fn get_cartesian(vec: &Vec4) -> Vec3 {
    canonicalize(vec).remove_row(3) //remove 'w'
}
