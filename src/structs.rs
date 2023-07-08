use std::borrow::Cow;

use nalgebra::{Vector3, SMatrix, Rotation3, Vector4, Point3};
use sdl2::pixels::Color;

pub type Vec3 = Vector3<f32>;
pub type Vec4 = Vector4<f32>;
pub type Mat3 = SMatrix<f32, 3, 3>;
pub type Mat4 = SMatrix<f32, 4, 4>;
pub type Rot3 = Rotation3<f32>;
pub type Plane = (Vec3, f32);

#[derive(Copy, Clone)]
pub struct PointC {
    pub point: Point<i32, i32>,
    pub h: f64,
}

#[derive(Clone, Default)]
pub struct Model {
    pub vertices: Vec<Point3<f32>>,
    pub triangles: Vec<(usize, usize, usize, Color)>,
}

#[derive(Copy, Clone)]
pub struct Point<T, C> {
    pub x: T,
    pub y: C,
}

pub struct Scene {
    pub camera: Camera,
    pub objects: Vec<Object>,
}

pub struct Camera {
    pub position: Vec3,
    pub rotation: Rot3,
    pub clipping_planes: Vec<Plane>,
}

#[derive(Clone)]
pub struct Transform {
    pub scale: f32,
    pub rotation: Rot3,
    pub position: Vec3,
}

#[derive(Clone)]
pub struct Object {
    pub model: &'static Model,
    pub transform: Transform,
    pub _render_model: Model,
    pub _bounding_sphere: BoundingSphere,
    //more...
}

#[derive(Default, Clone)]
pub struct BoundingSphere {
    pub center: Point3<f32>,
    pub r: f32,
}
//TRAITs & IMPLs
pub trait Reverse<T, C> {
    fn reverse(&self) -> Point<C, T> where C: Copy, T: Copy,;
}

impl<T, C> Reverse<T, C> for Point<T, C> {
    #[inline]
    /// Reverse the axes of Point<T, C>
    /// returns Point<C, T>
    /// s.t. T becomes dependent axis, and C becomes independent axis
    fn reverse(&self) -> Point<C, T>
    where
        C: Copy,
        T: Copy,
    {
        Point {x: self.y, y: self.x}
    }
}

pub trait ObjectConstructor {
    fn new(model: &'static Model, transform: Transform) -> Object;
}

pub trait PlaneConstructor {
    fn new(normal: Vec3, point: &Point3<f32>) -> Plane;
}

impl ObjectConstructor for Object {
    fn new(model: &'static Model, transform: Transform) -> Object {
        Object { 
            model,
            transform, 
            _render_model: Model::default(),
            _bounding_sphere: BoundingSphere::default(),
        }

    }
}

impl PlaneConstructor for Plane {
    fn new(normal: Vec3, point: &Point3<f32>) -> Plane {
        let product = normal.dot(&point.coords); //dot product
        //N*P + D = 0
        //D = -N*P
        (normal, -product)
    }
}