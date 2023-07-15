#![feature(convert_float_to_int)]

pub mod render;
pub mod structs;
pub mod timing;
pub mod mat;

use itertools::Itertools;
use mat::get_transform_mat;
use na::{Point3, distance};
use num_traits::Signed;
use structs::{Model, Object, Point, Rot3, Scene, Transform, Vec3, Camera, Mat4, Plane, DepthPoint, LightType, Triangle, Light, RenderTriangle};
use timing::TimerTrait;

use std::{time::Duration, collections::HashMap};

extern crate nalgebra as na;

use sdl2::{pixels::Color, event::Event, render::Canvas, video::Window, keyboard::Keycode};

use crate::structs::{ObjectConstructor, TriangleConstructor};

//CONSTS
pub const CANVAS_WIDTH: u32 = 800;
pub const CANVAS_HEIGHT: u32 = 800;
pub const BACKGROUND_COLOR: Color = Color::GRAY;//Color::WHITE;
pub const VIEWPORT_WIDTH: f32 = 1.0;
pub const VIEWPORT_HEIGHT: f32 = 1.0;
pub const VIEWPORT_Z_DIST: f32 = 0.5; //extend FOV so we can see the clipping happen on-canvas

pub fn main() -> Result<(), String> {
    let mut startup_timing = timing::new();
    startup_timing.start();
    let sdl = sdl2::init()?;
    let window = sdl.video()?.window("Rasterizer", crate::CANVAS_WIDTH, crate::CANVAS_HEIGHT)
        .position_centered()
        .opengl()
        .build()
        .map_err(|e| e.to_string())?;

    let mut canvas = window.into_canvas().build().map_err(|e| e.to_string())?;
    let mut scene = init_scene();
    let mut models: HashMap<String, Model> = HashMap::new();

    add_model("CUBE".to_string(), 
        vec![
            Point3::new(1., 1., 1.),
            Point3::new(-1., 1., 1.),
            Point3::new(-1., -1., 1.),
            Point3::new(1., -1., 1.),
            Point3::new(1., 1., -1.),
            Point3::new(-1., 1., -1.),
            Point3::new(-1., -1., -1.),
            Point3::new(1., -1., -1.),
        ], 
        None,
        vec![
            (1, 2, 0, Color::RED),
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
        &mut models,
        &scene.camera
    );
    add_model("DIAMOND".to_string(), 
        vec![
            Point3::new(0., 0., -1.), //0 BOTTOM POINT
            Point3::new(-1., 0., 0.), //1  LEFT POINT
            Point3::new(0., -1., 0.), //2  FORWARD POINT
            Point3::new(1., 0., 0.), //3 RIGHT POINT
            Point3::new(0., 1., 0.), //4  BACK POINT
            Point3::new(0., 0., 1.), //5  TOP POINT
        ],
        None,
        vec![
            (0, 1, 2, Color::RED),
            (0, 2, 3, Color::BLUE),
            (0, 3, 4, Color::GREEN),
            (0, 4, 1, Color::YELLOW),//bottom half

            (5, 2, 1, Color::BLUE),
            (5, 3, 2, Color::GREEN),
            (5, 4, 3, Color::YELLOW),
            (5, 4, 1, Color::RED), //top half
        ], 
        &mut models,
        &scene.camera
    );
    add_model("TRIANGLE".to_string(),
        vec![
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 0.0 ,0.0),
        ],
        None,
        vec![
            (1, 0, 2, Color::RED),
        ], 
        &mut models,
        &scene.camera,
    );

    scene.lights.push(Light {
        metadata: LightType::Ambient,
        intensity: Vec3::new(0.5, 0.5, 0.5),
    });
    scene.lights.push(Light {
        metadata: LightType::Point(Vec3::new(-5.0, 1.0, 2.0)),
        intensity: Vec3::new(0.5, 0.5, 0.5),
    });

    
    //println!("here");
    let base_pos = Vec3::new(0., -2., 5.0);

    for x in 0..3 {
        for z in 0..3 {
            //println!("cur obj: {x}, {y}, cur pos: {}", cur_pos.to_string());
            scene.objects.push(Object::new( 
                "CUBE".to_string(),
                Transform {
                    position: base_pos + Vec3::new(x as f32 * 2., z as f32 + x as f32, z as f32),
                    scale: 1.,
                    rotation:Rot3::from_euler_angles(0.5, -0.5, 0.),
                     //Rot3::from_axis_angle(&Vec3::y_axis(), 1.2),
                }
            ));
        }
    }
    /* 
    scene.objects.push(Object::new(
        "DIAMOND".to_string(),
        Transform { 
            scale: 1.,
            rotation: Rot3::from_axis_angle(&Vec3::z_axis(), 1.),
            position: Vec3::new(0., 0., 5.),
        }
    ));

    
    scene.objects.push(Object::new(
        "CUBE".to_string(),
        Transform {
            position: Vec3::new(1.4, 0.0, 10.0),
            rotation: Rot3::from_axis_angle(&Vec3::z_axis(), 0.),
            scale: 1.,
        },
    ));
    
    scene.objects.push(Object::new( 
        "CUBE".to_string(),
        Transform {
            position: Vec3::new(-1.0, 4.0, 9.0),
            rotation: Rot3::from_axis_angle(&Vec3::y_axis(), 1.2),
            scale: 0.5,
        },
    ));

    //TODO: why does this triangle always fail to be ordered?
    scene.objects.push(Object::new(
        "TRIANGLE".to_string(),
        Transform { 
            scale: 1.0,
            rotation: Rot3::from_axis_angle(&Vec3::x_axis(), 0.5), 
            position: Vec3::new(-1.0, 4.0, 7.0), 
        },
    ));
    
    scene.objects.push(Object::new(
        &MODEL_TRIANGLE,
        Transform { 
            scale: 1.0,
            rotation: Rot3::from_axis_angle(&Vec3::z_axis(), 0.), 
            position: Vec3::new(0.0, 5.0, 0.0),
        },
    )); */

    let mut event_pump = sdl.event_pump()?;
    let mut running: bool = true;
    let mut camera_y_rot: f32 = 0.;
    let mut trans_y_pos: f32 = 0.;

    let mut render_timing = timing::new();
    
    let mut cam_mat = compute_cam_matrix(&scene.camera);
    startup_timing.elapse();
    //let mut combined_proj_mat = scene_to_canvas_mat * cam_mat;
    while running {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit {..} => running = false,
                Event::KeyDown { timestamp, window_id, keycode, ..} => {
                    if keycode.is_some_and(|key| key == Keycode::Right) {
                        camera_y_rot += 0.02;
                    } else if keycode.is_some_and(|key| key == Keycode::Left) {
                        camera_y_rot -= 0.02;
                    } 
                    scene.camera.rotation = Rot3::from_axis_angle(&Vec3::y_axis(), -camera_y_rot);
                    cam_mat = compute_cam_matrix(&scene.camera);
                },
                _ => (),
            }
        }
        trans_y_pos += 0.01;


        for object in scene.objects.as_mut_slice() {
            object.transform.rotation = Rot3::from_euler_angles(trans_y_pos, trans_y_pos, trans_y_pos);
        }

        //scene.objects[0].transform.rotation = Rot3::from_axis_angle(&Vec3::x_axis(), trans_z_pos);
        
        //scene.objects[0].transform.position = Vec3::new(1.5, f32::cos(trans_y_pos), 10.);
        //scene.objects[0].transform.rotation = Rot3::from_axis_angle(&Vec3::x_axis(), trans_y_pos*2.);
        //scene.objects[1].transform.rotation = Rot3::from_axis_angle(&Vec3::y_axis(), trans_y_pos);
        //scene.objects[1].transform.position = Vec3::new(-1. + f32::cos(trans_y_pos), 0., 8. + 2.*f32::sin(trans_y_pos));
        
        //scene.lights[1].metadata = LightType::Point(Vec3::new(f32::sin(trans_y_pos), f32::cos(trans_y_pos), 0.0));
        // RENDER PIPELINE

        // project: transforms all model vertices into camera space (model -> world -> camera) and calculates
        // general bounding sphere of object (for clipping detection)

        // clip: uses the projected vertices to determine what can and can't be seen by the camera (what is not within clipping volume)
        // and purges or creates new vertices that are within the bounds of the camera.
        // this also includes back face culling (check if normal > 90 degrees from camera POV, s.t. the face is completely hidden)

        // render: takes the vertices that are left from the objects, performs (camera -> viewport -> canvas)
        // and renders wireframes for each triange/vertex set

        render_timing.start();
        //project all vertices, and calculates bounding spheres
        project_scene_objects(&mut scene, &models, cam_mat);

        //clip scene s.t. objects, or vertices in partially off-screen objects are not rendered
        clip_scene_objects(&mut scene);

        //back face culling s.t. vertices facing away from camera are not rendered
        back_face_culling(&mut scene);

        //project and render visible vertices onto the canvas, THIS ALSO HANDLES LIGHTING!
        render_scene_objects(&mut canvas, &scene);
        render_timing.elapse();

        //limit thread speed
        std::thread::sleep(Duration::from_millis(1));
    }
    println!("RENDER TIMING\n{}", render_timing.report());
    println!("STARTUP TIMING\n{}", startup_timing.report());
    Ok(())
}

fn add_model(name: String, mut vertices: Vec<Point3<f32>>, input_normals: Option<Vec<Vec3>>, _triangles: Vec<(usize, usize, usize, Color)>, map: &mut HashMap<String, Model>, camera: &Camera) { 
    let generate_normals: bool;
    let mut normals: Vec<Vec3>;
    let null_vec = Vec3::new(-100., -100., -100.);
    if input_normals.is_some() {
        normals = input_normals.unwrap();
        assert_eq!(vertices.len(), normals.len(), "input normals and vertices are not the same length!");
        generate_normals = false;
    } else {
        normals = vec![null_vec; vertices.len()];
        generate_normals = true;
    }

    let mut triangles = Vec::new();
    for i in 0.._triangles.len() {
        let triangle = _triangles[i];
        let mut vert_order = [triangle.0, triangle.1, triangle.2];

        let (normal, success) = make_clockwise(&vertices, &mut vert_order, camera);

        if !success {
            println!("model {name} contains triangle {:?} that could not be put in CW orientation.", vert_order);
        }

        if generate_normals {
            for i in 0..3 {
                let cur_vertex = vert_order[i];
                if normals[cur_vertex] == null_vec || normals[cur_vertex] == normal {
                    normals[cur_vertex] = normal;
                } else { //corner vertex, we must duplicate for each face
                    let index = vertices.len();
                    vertices.push(vertices[cur_vertex]); //duplicate
                    normals.push(normal);
                    vert_order[i] = index;
                }
            }
        }
        triangles.push(Triangle {
            index: (vert_order[0], vert_order[1], vert_order[2]),
            color: triangle.3,
            normal,
        });
    }
    let mut zipped_vertices: Vec<(Point3<f32>, Vec3)> = Vec::new();
    for i in 0..vertices.len() {
        zipped_vertices.push((vertices[i], normals[i]));
    }

    let model = Model {
        vertices: zipped_vertices,
        triangles,
    };
    //println!("{:?}", model.triangles);
    map.insert(name, model);
}

fn project_scene_objects(scene: &mut Scene, models: &HashMap<String, Model>, cam_mat: Mat4) {
    for object in scene.objects.as_mut_slice() {
        let object_model = models.get(&object.model).unwrap();
        let model_transform_mat = cam_mat * get_transform_mat(&object.transform);
        let mut model = Model {
            vertices: Vec::new(),
            triangles: object_model.triangles.clone(),
        };
        let mut average_center: Point3<f32> = Point3::origin();
        for (v, normal) in object_model.vertices.as_slice() {
            let t_vertex  = model_transform_mat * v.to_homogeneous();
            let vertex = Point3::from_homogeneous(t_vertex).unwrap();

            let t_normal = model_transform_mat * normal.to_homogeneous();
            let n_normal = Vec3::from_homogeneous(t_normal).unwrap();
            average_center.x += vertex.x;
            average_center.y += vertex.y;
            average_center.z += vertex.z;
            model.vertices.push((vertex, n_normal)); 
        }
        average_center /= model.vertices.len() as f32;
        let mut largest_distance = 0.;
        for (v, _) in &model.vertices {
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

fn make_clockwise(vertices: &Vec<Point3<f32>>, order: &mut [usize; 3], camera: &Camera) -> (Vec3, bool) {
    let (mut normal, mut is_cw) = is_clockwise(vertices, order, camera);
    if is_cw {
        return (normal, true)
    }
    for comb in order.iter().permutations(3) {
        let cur_comb = [*comb[0], *comb[1], *comb[2]];
        (normal, is_cw) = is_clockwise(vertices, &cur_comb, camera);
        if is_cw {
            *order = cur_comb;
            return (normal, true)
        }
    }
    (normal, false)
}

fn is_clockwise(vertices: &Vec<Point3<f32>>, order: &[usize; 3], camera: &Camera) -> (Vec3, bool) {
    let a = vertices[order[0]].coords;
    let b = vertices[order[1]].coords; 
    let c = vertices[order[2]].coords;
    
    let normal = (b - a).cross(&(c - a));
    (normal, normal.dot(&(a -camera.position)) < 0.)
}

fn render_scene_objects(canvas: &mut Canvas<Window>, scene: &Scene) {
    let mut zbuffer = vec![0_f32; (CANVAS_WIDTH * CANVAS_HEIGHT) as usize];
    
    canvas.set_draw_color(crate::BACKGROUND_COLOR); //clear out bg
    canvas.clear();

    for object in scene.objects.as_slice() { //iterate through all scene objects, render out to screen
        render_object(scene, canvas, object, &mut zbuffer);
    }
    canvas.present();
}

fn clip_scene_objects(scene: &mut Scene) {
    for object in scene.objects.as_mut_slice() {
        if clip_model(object, &scene.camera.clipping_planes) {
            object._render_model = Model::default();
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
    let mut clipped_triangles: Vec<Triangle> = Vec::new();
    for triangle in &object._render_model.triangles {
        clip_triangle(triangle, &mut object._render_model.vertices, plane, &mut clipped_triangles);
    }
    object._render_model.triangles = clipped_triangles;
    return object._render_model.triangles.len() == 0; 
    //tells render to use 'default' (empty) model if no triangles exist in the clipping volume
}

fn clip_triangle(triangle: &Triangle, vertices: &mut Vec<(Point3<f32>, Vec3)>, plane: &Plane, clipped_triangles: &mut Vec<Triangle>) {
    let d0 = signed_distance(plane, vertices[triangle.index.0].0);
    let d1 = signed_distance(plane, vertices[triangle.index.1].0);
    let d2 = signed_distance(plane, vertices[triangle.index.2].0);

    // either entirely inside bounds or entirely outside bounds
    if d0.is_positive() && d1.is_positive() && d2.is_positive() {
        clipped_triangles.push(*triangle);
        return;
    } else if d0.is_negative() && d1.is_negative() && d2.is_negative() {
        return;
    }

    // partially in bounds
    let mut ordered_vertices = vec![(d0, triangle.index.0), (d1, triangle.index.1), (d2, triangle.index.2)];
    ordered_vertices.sort_by(|a, b| b.0.total_cmp(&a.0));
    let a = ordered_vertices[0]; //definitely positive
    let b = ordered_vertices[1]; //could be positive
    let c = ordered_vertices[2]; //not positive

    let AC = (&vertices[a.1].0.coords, &vertices[c.1].0.coords);
    let AB = (&vertices[a.1].0.coords, &vertices[b.1].0.coords);
    let BC = (&vertices[b.1].0.coords, &vertices[c.1].0.coords);
    if a.0.is_positive() && b.0.is_positive() { //2 vertices inside, one out
        let a_prime = plane_intersection(plane, AC);
        let b_prime = plane_intersection(plane, BC);
        let aprime_vertex_pos = vertices.len();
        vertices.push((a_prime, triangle.normal));
        let bprime_vertex_pos = vertices.len();
        vertices.push((b_prime, triangle.normal));
        clipped_triangles.push(Triangle::new(a.1, b.1, aprime_vertex_pos, triangle.color)); //A, B, A'
        clipped_triangles.push(Triangle::new(a.1, b.1, bprime_vertex_pos, triangle.color)); //A, B, B'
        return;
    } else if a.0.is_positive() {
        let b_prime = plane_intersection(plane, AB);
        let c_prime = plane_intersection(plane, AC);
        let bprime_vertex_pos = vertices.len();
        vertices.push((b_prime, triangle.normal));
        let cprime_vertex_pos = vertices.len();
        vertices.push((c_prime, triangle.normal));
        clipped_triangles.push(Triangle::new(a.1, bprime_vertex_pos, cprime_vertex_pos, triangle.color));
        return;
    }
    
}

fn back_face_culling(scene: &mut Scene) {
    let camera = &scene.camera;

    for object in &mut scene.objects {
        let mut retained_triangles = Vec::new();
        let vertices = &object._render_model.vertices;
        for triangle in object._render_model.triangles.as_mut_slice() {
            let cam_vec = vertices[triangle.index.0].0.coords - camera.position;
            triangle.normal = 
                (vertices[triangle.index.1].0 - vertices[triangle.index.0].0)
                .cross(&(vertices[triangle.index.2].0 - vertices[triangle.index.0].0)); // B-A x C-A
            if triangle.normal.dot(&cam_vec) > 0. { //> 0. is front facing 
                retained_triangles.push(*triangle);
            }
        }
        object._render_model.triangles = retained_triangles;
    }

}

fn plane_intersection(plane: &Plane, line: (&Vec3, &Vec3)) -> Point3<f32> {
    let line_vec = line.1 - line.0;
    let t = (-plane.1 - plane.0.dot(line.0)) / plane.0.dot(&line_vec);
    Point3::from(line.0 + t*line_vec)
}

fn signed_distance(plane: &Plane, point: Point3<f32>) -> f32 {
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
fn render_object(scene: &Scene, canvas: &mut Canvas<Window>, object: &Object, zbuf: &mut Vec<f32>) {
    let mut projected_vertices = Vec::new();

    //project from camera space -> viewport view -> canvas coordinate
    for (vertex, normal) in object._render_model.vertices.as_slice() {
        projected_vertices.push(project_vertex(*vertex, *normal));
    }

    //println!("render triangles from vertices pushed");
    for triangle in object._render_model.triangles.as_slice() {
        let v0 = triangle.index.0;
        let v1 = triangle.index.1;
        let v2 = triangle.index.2;
        let render_triangle = RenderTriangle {
            points: (projected_vertices[v0], projected_vertices[v1], projected_vertices[v2]),
            specular: 0,
            color: triangle.color,
        };
        render::draw_triangle(scene, canvas, zbuf, &render_triangle);
    }
    //println!("finished");
}

fn init_scene() -> Scene {
    let one_over_sqrt2 = 1./f32::sqrt(2.);
    let clipping_planes = vec![ //90 degree POV kind of
        (Vec3::z(), -VIEWPORT_Z_DIST), //near
        (Vec3::new(one_over_sqrt2, 0., one_over_sqrt2), 0.), //left
        (Vec3::new(-one_over_sqrt2, 0., one_over_sqrt2), 0.), //right
        (Vec3::new(0., one_over_sqrt2, one_over_sqrt2), 0.), //bottom
        (Vec3::new(0., -one_over_sqrt2, one_over_sqrt2), 0.), //top
    ];

    Scene {
        camera: Camera { 
            position: Vec3::zeros(), 
            rotation: Rot3::from_axis_angle(&Vec3::z_axis(), 0.),
            clipping_planes,
        },
        objects: Vec::new(),
        lights: Vec::new(),
    }
}

/// convert viewport (x,y) 'real' values into canvas pixel values
#[inline]
fn viewport_to_canvas(x: f32, y: f32, z: f32, normal: Vec3) -> DepthPoint {
    DepthPoint {
        point: Point {
            x: (x * (crate::CANVAS_WIDTH as f32/crate::VIEWPORT_WIDTH)).round() as i32,
            y: (y * (crate::CANVAS_HEIGHT as f32/crate::VIEWPORT_HEIGHT)).round() as i32,
        },
        z,
        normal,
    }
}

#[inline]
fn project_vertex(vertex: Point3<f32>, normal: Vec3) -> DepthPoint {
    viewport_to_canvas(
        vertex.x * crate::VIEWPORT_Z_DIST / vertex.z,
        vertex.y * crate::VIEWPORT_Z_DIST / vertex.z,
        vertex.z,
        normal,
    ) //make use of similar triangles to project vertices
}
