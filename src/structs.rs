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

#[derive(Copy, Clone)]
pub struct DepthPoint {
    pub point: Point<i32, i32>,
    pub normal: Vec3,
    pub z: f32,
}

#[derive(Copy, Clone)]
pub struct RenderTriangle {
    pub points: (DepthPoint, DepthPoint, DepthPoint),
    pub specular: u32,
    pub color: Color,
    //pub texture: ,
}

#[derive(Clone, Default)]
pub struct Model {
    pub vertices: Vec<(Point3<f32>, Vec3)>,
    pub triangles: Vec<Triangle>,
}

#[derive(Clone, Copy)]
pub struct Triangle {
    pub index: (usize, usize, usize),
    pub color: Color,
    pub normal: Vec3,
}

#[derive(Copy, Clone)]
pub struct Point<T, C> {
    pub x: T,
    pub y: C,
}

pub struct Scene {
    pub camera: Camera,
    pub objects: Vec<Object>,
    pub lights: Vec<Light>,
}

pub struct Light {
    pub metadata: LightType,
    pub intensity: Vec3,
}

#[derive(PartialEq)]
pub enum LightType {
    Directional(Vec3), //Light direction VECTOR
    Point(Vec3),       //Light position VECTOR
    Ambient,
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
    pub model: String,
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
    fn new(model: String, transform: Transform) -> Object;
}

pub trait PlaneConstructor {
    fn new(normal: Vec3, point: &Point3<f32>) -> Plane;
}

pub trait TriangleConstructor {
    fn new(a: usize, b: usize, c: usize, color: Color) -> Triangle;
}

impl ObjectConstructor for Object {
    fn new(model: String, transform: Transform) -> Object {
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

impl TriangleConstructor for Triangle {
    fn new(a: usize, b: usize, c: usize, color: Color) -> Triangle {
        Triangle {
            index: (a, b, c),
            color,
            normal: Vec3::zeros(),
        }
    }
}