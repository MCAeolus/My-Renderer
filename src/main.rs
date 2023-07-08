#![feature(convert_float_to_int)]

pub mod render;
pub mod structs;
pub mod timing;
pub mod mat;

use lazy_static::lazy_static;
use mat::get_transform_mat;
use na::{Matrix3x4, Point3, distance};
use num_traits::Signed;
use structs::{Model, Object, Point, Rot3, Scene, Transform, Vec3, Camera, Mat4, Plane};
use timing::TimerTrait;

use std::time::Duration;

extern crate nalgebra as na;

use sdl2::{pixels::Color, event::Event, render::Canvas, video::Window, keyboard::Keycode};

use crate::structs::ObjectConstructor;

pub const CANVAS_WIDTH: u32 = 800;
pub const CANVAS_HEIGHT: u32 = 800;
pub const BACKGROUND_COLOR: Color = Color::GRAY;//Color::WHITE;
pub const VIEWPORT_WIDTH: f32 = 1.0;
pub const VIEWPORT_HEIGHT: f32 = 1.0;
pub const VIEWPORT_Y_DIST: f32 = 0.5; //extend FOV so we can see the clipping happen on-canvas


/*
MAT MP
| D * CW/VW, 0        , 0, 0,
| 0        , D * CH/VH, 0, 0,
| 0        , 0        , 1, 0,
*/
const PROJECTION_MATRIX: Matrix3x4<f32> = Matrix3x4::new( //3d-canvas & perspective projection mat MP 
    VIEWPORT_Y_DIST * CANVAS_WIDTH as f32 / VIEWPORT_WIDTH, 0., 0., 0.,
    0., VIEWPORT_Y_DIST * CANVAS_HEIGHT as f32 / VIEWPORT_HEIGHT, 0., 0.,
    0., 0., 1., 0.,
);

//instantiate models
lazy_static! {

    static ref MODEL_CUBE: Model = Model {
        vertices: vec![
            Point3::new(1., 1., 1.),
            Point3::new(-1., 1., 1.),
            Point3::new(-1., -1., 1.),
            Point3::new(1., -1., 1.),
            Point3::new(1., 1., -1.),
            Point3::new(-1., 1., -1.),
            Point3::new(-1., -1., -1.),
            Point3::new(1., -1., -1.),
        ],
        triangles: vec![
            (0, 1, 2, Color::RED),
            (0, 2, 3, Color::RED),
            (4, 0, 3, Color::GREEN),
            (4, 3, 7, Color::GREEN),
            (5, 4, 7, Color::BLUE),
            (5, 7, 6, Color::BLUE),
            (1, 5, 6, Color::YELLOW),
            (1, 6, 2, Color::YELLOW),
            (4, 5, 1, Color::from((159, 43, 104))),
            (4, 1, 0, Color::from((159, 43, 104))),
            (2, 6, 7, Color::CYAN),
            (2, 7, 3, Color::CYAN),
        ],
    };
    static ref MODEL_DIAMOND: Model = Model {
        vertices: vec![
            Point3::new(0., 0., -1.), //0 BOTTOM POINT
            Point3::new(-1., 0., 0.), //1  LEFT POINT
            Point3::new(0., -1., 0.), //2  FORWARD POINT
            Point3::new(1., 0., 0.), //3 RIGHT POINT
            Point3::new(0., 1., 0.), //4  BACK POINT
            Point3::new(0., 0., 1.), //5  TOP POINT
        ],
        triangles: vec![
            (0, 1, 2, Color::RED),
            (0, 2, 3, Color::BLUE),
            (0, 3, 4, Color::GREEN),
            (0, 4, 1, Color::YELLOW),//bottom half

            (5, 2, 1, Color::BLUE),
            (5, 3, 2, Color::GREEN),
            (5, 4, 3, Color::YELLOW),
            (5, 4, 1, Color::RED), //top half
        ],
    };
    static ref MODEL_TRIANGLE: Model = Model {
        vertices: vec![
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 0.0),
        ],
        triangles: vec![
            (0, 1, 2, Color::GREEN),
        ],
    };
}

//CONSTs

pub fn main() -> Result<(), String> {
    let sdl = sdl2::init()?;
    let window = sdl.video()?.window("Rasterizer", crate::CANVAS_WIDTH, crate::CANVAS_HEIGHT)
        .position_centered()
        .opengl()
        .build()
        .map_err(|e| e.to_string())?;

    let mut canvas = window.into_canvas().build().map_err(|e| e.to_string())?;
    let mut scene = init_scene();

    //let mut cur_pos = Vec3::new(-10., -10., -0.1);

    /*for x in 0..20 {
        for y in 0..20 {
            println!("cur obj: {x}, {y}, cur pos: {}", cur_pos.to_string());
            scene.objects.push(Object {
                model: &MODEL_CUBE,
                transform: Transform {
                    position: cur_pos,
                    scale: 1.,
                    rotation: Rot3::from_axis_angle(&Vec3::z_axis(), 0.),
                }
            });
            cur_pos += Vec3::y();
        }
        cur_pos += Vec3::new(1., -cur_pos.y, 0.);
    }*/

    scene.objects.push(Object::new(
        &MODEL_DIAMOND,
        Transform { 
            scale: 1.,
            rotation: Rot3::from_axis_angle(&Vec3::z_axis(), 1.),
            position: Vec3::new(0., 5., 0.),
        }
    ));

    
    scene.objects.push(Object::new(
        &MODEL_CUBE,
        Transform {
            position: Vec3::new(1.4, 10.0, 0.0),
            rotation: Rot3::from_axis_angle(&Vec3::z_axis(), 0.),
            scale: 1.,
        },
    ));
    
    scene.objects.push(Object::new( 
        &MODEL_CUBE,
        Transform {
            position: Vec3::new(-1.0, 9.0, 4.0),
            rotation: Rot3::from_axis_angle(&Vec3::y_axis(), 1.2),
            scale: 0.5,
        },
    ));

    scene.objects.push(Object::new(
        &MODEL_CUBE,
        Transform { 
            scale: 1.0,
            rotation: Rot3::from_axis_angle(&Vec3::x_axis(), 0.5), 
            position: Vec3::new(-1.0, 7.0, 4.0), 
        },
    ));
/*
    scene.objects.push(Object::new(
        &MODEL_TRIANGLE,
        Transform { 
            scale: 1.0,
            rotation: Rot3::from_axis_angle(&Vec3::z_axis(), 0.), 
            position: Vec3::new(0.0, 5.0, 0.0),
        },
    ));*/

    let mut event_pump = sdl.event_pump()?;
    let mut running: bool = true;
    let mut camera_z_rot: f32 = 0.;
    let mut trans_z_pos: f32 = 0.;

    let mut render_timing = timing::new();
    
    let mut cam_mat = compute_cam_matrix(&scene.camera);
    //let mut combined_proj_mat = scene_to_canvas_mat * cam_mat;
    while running {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit {..} => running = false,
                Event::KeyDown { timestamp, window_id, keycode, ..} => {

                    if keycode.is_some_and(|key| key == Keycode::Right) {
                        camera_z_rot += 0.02;
                    } else if keycode.is_some_and(|key| key == Keycode::Left) {
                        camera_z_rot -= 0.02;
                    } 
                    scene.camera.rotation = Rot3::from_axis_angle(&Vec3::z_axis(), camera_z_rot);
                    //println!("{:?}", scene.camera.rotation.angle());
                    cam_mat = compute_cam_matrix(&scene.camera);
                    //combined_proj_mat = scene_to_canvas_mat * cam_mat;
                },
                _ => (),
            }
        }
        trans_z_pos += 0.01;

        //scene.objects[0].transform.rotation = Rot3::from_axis_angle(&Vec3::x_axis(), trans_z_pos);
        
        scene.objects[0].transform.position = Vec3::new(1.5, 10., f32::cos(trans_z_pos));
        scene.objects[0].transform.rotation = Rot3::from_axis_angle(&Vec3::x_axis(), trans_z_pos*2.);
        scene.objects[1].transform.rotation = Rot3::from_axis_angle(&Vec3::z_axis(), trans_z_pos);
        scene.objects[1].transform.position = Vec3::new(-1. + f32::cos(trans_z_pos), 8. + 2.*f32::sin(trans_z_pos), 0.);
        
        // RENDER PIPELINE
        // project: transforms all model vertices into camera space (model -> world -> camera) and calculates
        // general bounding sphere of object (for clipping detection)

        // clip: uses the projected vertices to determine what can and can't be seen by the camera (what is not within clipping volume)
        // and purges or creates new vertices that are within the bounds of the camera.

        // render: takes the vertices that are left from the objects, performs (camera -> viewport -> canvas)
        // and renders wireframes for each triange/vertex set
        // TODO! render should also depth sort the vertices before rendering
        // (so render furthest away vertices first, and closest last)

        render_timing.start();
        //project all vertices, and calculates bounding spheres
        project_scene_objects(&mut scene, cam_mat);

        //clip scene 
        clip_scene_objects(&mut scene);

        //render visible vertices
        render_scene_objects(&mut canvas, &scene);
        render_timing.elapse();

        //limit thread speed
        std::thread::sleep(Duration::from_millis(1));
    }
    println!("RENDER TIMING\n{}", render_timing.report());
    Ok(())
}

fn project_scene_objects(scene: &mut Scene, cam_mat: Mat4) {
    for object in scene.objects.as_mut_slice() {
        let model_transform_mat = cam_mat * get_transform_mat(&object.transform);
        let mut model = Model {
            vertices: Vec::new(),
            triangles: object.model.triangles.clone(),
        };
        let mut average_center: Point3<f32> = Point3::origin();
        for v in object.model.vertices.as_slice() {
            let t_vertex  = model_transform_mat * v.to_homogeneous();
            let vertex = Point3::from_homogeneous(t_vertex).unwrap();
            average_center.x += vertex.x;
            average_center.y += vertex.y;
            average_center.z += vertex.z;
            model.vertices.push(vertex); 
        }
        average_center /= model.vertices.len() as f32;
        let mut largest_distance = 0.;
        for v in &model.vertices {
            let cur_distance = distance(v, &average_center);
            if cur_distance > largest_distance {
                largest_distance = cur_distance;
            }
        }
        object._bounding_sphere.center = average_center;
        object._bounding_sphere.r = largest_distance;
        object._render_model = model;
    }
}


fn render_scene_objects(canvas: &mut Canvas<Window>, scene: &Scene) {
    
    canvas.set_draw_color(crate::BACKGROUND_COLOR); //clear out bg
    canvas.clear();

    //perform a rudimentary depth sort
    //TODO!this should be done by vertex, not by object.
    let mut object_order: Vec<usize> = (0..scene.objects.len()).collect();

    // sort by y val (greatest to least)
    object_order.sort_by(|a, b| 
        scene.objects[*b]._bounding_sphere.center.y.total_cmp(&scene.objects[*a]._bounding_sphere.center.y));

    for object_index in object_order { //iterate through all scene objects, render out to screen
        render_object(canvas, &scene.objects[object_index]);
    }
    
    //let center_axis_draw = Vec3::new(0., 5., 0.);

    //y-axis line RED
    /*
    draw_line(
        canvas, 
        project_vertex(center_axis_draw), 
        project_vertex(center_axis_draw + Vec3::y()), 
        Color::RED
    ); 
    //x-axis line BLUE
    draw_line(
        canvas,
        project_vertex(center_axis_draw),
        project_vertex(center_axis_draw + Vec3::x()),
        Color::BLUE
    );
    //z-axis line GREEN
    draw_line(
        canvas,
        project_vertex(center_axis_draw),
        project_vertex(center_axis_draw + Vec3::z()),
        Color::GREEN
    );*/

    canvas.present();
}

fn clip_scene_objects(scene: &mut Scene) {

    for object in scene.objects.as_mut_slice() {
        if clip_model(object, &scene.camera.clipping_planes) {
            object._render_model = Model::default();
            //println!("clip says model out of bounds");
        }
    }
}

fn clip_model(object: &mut Object, planes: &Vec<Plane>) -> bool {
    for plane in planes {
        let outside_plane = clip_object_by_plane(object, plane);
        if outside_plane {
            return true;
        }
    }
    return false;
}

fn clip_object_by_plane(object: &mut Object, plane: &Plane) -> bool {
    let distance = signed_distance(plane, object._bounding_sphere.center);
    let r = object._bounding_sphere.r;
    if distance > r { 
        return false;
    } else if distance < -r {
        return true;
    } else {
        return clip_triangles_by_plane(object, plane);
    }
}

fn clip_triangles_by_plane(object: &mut Object, plane: &Plane) -> bool {
    let mut clipped_triangles: Vec<(usize, usize, usize, Color)> = Vec::new();
    for triangle in &object._render_model.triangles {
        clip_triangle(triangle, &mut object._render_model.vertices, plane, &mut clipped_triangles);
    }
    object._render_model.triangles = clipped_triangles;
    return object._render_model.triangles.len() == 0; 
    //tells render to use 'default' (empty) model if no triangles exist in the clipping volume
}

fn clip_triangle(triangle: &(usize, usize, usize, Color), vertices: &mut Vec<Point3<f32>>, plane: &Plane, clipped_triangles: &mut Vec<(usize, usize, usize, Color)>) {
    let d0 = signed_distance(plane, vertices[triangle.0]);
    let d1 = signed_distance(plane, vertices[triangle.1]);
    let d2 = signed_distance(plane, vertices[triangle.2]);

    // either entirely inside bounds or entirely outside bounds
    if d0.is_positive() && d1.is_positive() && d2.is_positive() {
        clipped_triangles.push(*triangle);
        return;
    } else if d0.is_negative() && d1.is_negative() && d2.is_negative() {
        return;
    }

    // partially in bounds
    let mut ordered_vertices = vec![(d0, triangle.0), (d1, triangle.1), (d2, triangle.2)];
    ordered_vertices.sort_by(|a, b| b.0.total_cmp(&a.0));
    let a = ordered_vertices[0]; //definitely positive
    let b = ordered_vertices[1]; //could be positive
    let c = ordered_vertices[2]; //not positive

    let AC = (&vertices[a.1].coords, &vertices[c.1].coords);
    let AB = (&vertices[a.1].coords, &vertices[b.1].coords);
    let BC = (&vertices[b.1].coords, &vertices[c.1].coords);
    if a.0.is_positive() && b.0.is_positive() { //2 vertices inside, one out
        let a_prime = plane_intersection(plane, AC);
        let b_prime = plane_intersection(plane, BC);
        let aprime_vertex_pos = vertices.len();
        vertices.push(a_prime);
        let bprime_vertex_pos = vertices.len();
        vertices.push(b_prime);
        clipped_triangles.push((a.1, b.1, aprime_vertex_pos, triangle.3)); //A, B, A'
        clipped_triangles.push((a.1, b.1, bprime_vertex_pos, triangle.3)); //A, B, B'
        return;
    } else if a.0.is_positive() {
        let b_prime = plane_intersection(plane, AB);
        let c_prime = plane_intersection(plane, AC);
        let bprime_vertex_pos = vertices.len();
        vertices.push(b_prime);
        let cprime_vertex_pos = vertices.len();
        vertices.push(c_prime);
        clipped_triangles.push((a.1, bprime_vertex_pos, cprime_vertex_pos, triangle.3));
        return;
    }
    
}

fn plane_intersection(plane: &Plane, line: (&Vec3, &Vec3)) -> Point3<f32> {
    let line_vec = line.1 - line.0;
    let t = (-plane.1 - plane.0.dot(line.0)) / plane.0.dot(&line_vec);
    Point3::from(line.0 + t*line_vec)
}

fn signed_distance (plane: &Plane, point: Point3<f32>) -> f32 {
    let normal = plane.0;
    (point.x * normal.x + point.y * normal.y + point.z * normal.z) + plane.1
}

fn compute_cam_matrix(camera: &Camera) -> Mat4 {
    let mut mat = camera.rotation.to_homogeneous().try_inverse().unwrap();
    mat *= Mat4::new_translation(&camera.position).try_inverse().unwrap();
    mat
}

//we make a set of vertices with indexes 0..n, and triangles, which point to the vertices
// i.e.
//vertices: 0=(1.0, 2.0, 4.0), 1, 2, ...
//triangles: 0=(v1, v3, v2, color:red)...
fn render_object(canvas: &mut Canvas<Window>, object: &Object) {
    
    let mut projected_vertices = Vec::new();

    //project from camera space -> viewport view -> canvas coordinate
    for vertex in object._render_model.vertices.as_slice() {
        projected_vertices.push(project_vertex(*vertex));
    }

    //println!("render triangles from vertices pushed");
    for (v0, v1, v2, color) in object._render_model.triangles.as_slice() {
        render::draw_wireframe_triangle(canvas, projected_vertices[*v0], projected_vertices[*v1], projected_vertices[*v2], *color);
    }
    //println!("finished");
}

fn init_scene() -> Scene {
    let one_over_sqrt2 = 1./f32::sqrt(2.);
    let clipping_planes = vec![ //90 degree POV 
        (Vec3::y(), -VIEWPORT_Y_DIST), //near
        (Vec3::new(one_over_sqrt2, one_over_sqrt2, 0.), 0.), //left
        (Vec3::new(-one_over_sqrt2, one_over_sqrt2, 0.), 0.), //right
        (Vec3::new(0., one_over_sqrt2, one_over_sqrt2), 0.), //bottom
        (Vec3::new(0., one_over_sqrt2, -one_over_sqrt2), 0.), //top
    ];

    Scene {
        camera: Camera { 
            position: Vec3::zeros(), 
            rotation: Rot3::from_axis_angle(&Vec3::z_axis(), 0.),
            clipping_planes: clipping_planes,
        },
        objects: Vec::new(),
    }
}

/// convert viewport (x,y) 'real' values into canvas pixel values
#[inline]
fn viewport_to_canvas(x: f32, y: f32) -> Point<i32, i32> {
    Point {
        x: (x * (crate::CANVAS_WIDTH as f32/crate::VIEWPORT_WIDTH)).round() as i32,
        y: (y * (crate::CANVAS_HEIGHT as f32/crate::VIEWPORT_HEIGHT)).round() as i32,
    }
}

#[inline]
fn project_vertex(vertex: Point3<f32>) -> Point<i32, i32> {
    viewport_to_canvas(
        vertex.x * crate::VIEWPORT_Y_DIST / vertex.y,
        vertex.z * crate::VIEWPORT_Y_DIST / vertex.y
    ) //make use of similar triangles to project vertices
}
