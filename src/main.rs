pub mod timing;

use std::f32::consts::PI;

use ndarray::{arr2, Array2, OwnedRepr, Ix3 };

//CHAPTER 3 - updated to incorporate lighting types
use sdl2::{pixels::Color, rect::Point, event::Event, libc::{close, COPYFILE_RECURSE_DIR}};

use crate::timing::TimerTrait;

//TYPEDEFs
type Vec3 = (f32, f32, f32);

//STRUCTs
struct Scene {
    spheres: Vec<Sphere>,
    lights: Vec<Light>
}

struct Sphere {
    color: Color,
    position: Vec3,
    radius: f32,
    specular: u32, // TODO: this should be a general object/surface value
    reflective: f32, // how reflective the surface is, should be in range [0,1]
}

struct Light {
    metadata: LightType,
    intensity: Vec3,
}

struct Camera {
    position: Vec3,
    orientation: Array2<f32>
}

//ENUMs
#[derive(PartialEq)]
enum LightType {
    Directional(Vec3), //Light direction VECTOR
    Point(Vec3),       //Light position VECTOR
    Ambient,
}

//CONSTs
const INFINITY: i32 = 1000; //"Large" number for now

const VIEWPORT_WIDTH: i32 = 1;
const VIEWPORT_HEIGHT: i32 = 1;

const CANVAS_WIDTH: u32 = 800;
const CANVAS_HEIGHT: u32 = 800;

const EPSILON: f32 = 0.001;
const TRACE_RECURSION_DEPTH: u8 = 3;

//this is renderer V1, we use hard-coded camera position
const DEFAULT_CAMERA_POSITION: Vec3 = (0., 0., 0.); //origin of world
const BACKGROUND_COLOR: Color = Color::RGB(0, 0, 0);

pub fn main() -> Result<(), String> {
    let sdl = sdl2::init()?;
    let window = sdl.video()?.window("Renderer", CANVAS_WIDTH, CANVAS_HEIGHT)
        .position_centered()
        .opengl()
        .build()
        .map_err(|e| e.to_string())?;

    let mut canvas = window.into_canvas().build().map_err(|e| e.to_string())?;

    canvas.set_draw_color(BACKGROUND_COLOR);
    canvas.clear(); //set bg to be black

    //canvas.set_draw_color(Color::RGB(255,0,0));
    //canvas.draw_line(Point::new(0,0),  Point::new(100,100)).unwrap();

    //define scene
    let mut scene = Scene {
        spheres: Vec::new(),
        lights: Vec::new(),
    };

    let mut camera = Camera {
        position: (0., 0., 0.),
        orientation: z_rot_mat(180. * (PI / 180.)), //
    };

    //define spheres in scene
    scene.spheres.insert(0, Sphere {
        color: Color::WHITE,
        position: (-1., 3.5, 0.),
        radius: 1.,
        specular: 100,
        reflective: 0.2,
    });
    scene.spheres.insert(1, Sphere {
        color: Color::BLUE,
        position: (1., 3.5, -0.5),
        radius: 1.,
        specular: 500,
        reflective: 0.1,
    });
    scene.spheres.insert(2, Sphere {
        color: Color::GREEN,
        position: (0., 2.5, 1.),
        radius: 0.5,
        specular: 100,
        reflective: 0.5,
    });
    scene.spheres.insert(3, Sphere { //the "floor"
        color: Color::GRAY,
        position: (0., 0., -5001.),
        radius: 5000.,
        specular: 1000,
        reflective: 0.5,
    });

    //making spheres for each light source
    //todo: debug option that shows spheres for each light source
    scene.spheres.insert(4, Sphere {
        color: Color::WHITE,
        position: ((0.0, 0.0, 0.8)),
        radius: 0.1,
        specular: 10000,
        reflective: 0.,
    });
    scene.spheres.insert(5, Sphere {
        color: Color::WHITE,
        position: ((-7.0, 7.0, 0.0)),
        radius: 0.1,
        specular: 10000,
        reflective: 0.,
    });

    
    //define lighting elements
    //note: all light intensities are normalized so that the sum of each channel is 1
    scene.lights.insert(0, Light {
        metadata: LightType::Point((0.0, 0.0, 8.0)),
        intensity: (0.9, 0.9, 0.9), //color channels added, red point light
    });
    scene.lights.insert(1, Light {
        metadata: LightType::Point((-3.0, 3.0, 0.0)),
        intensity: v_scmult(&color_to_float_channels(&Color::BLUE), 1./255.),
    });
    scene.lights.insert(2, Light {
        metadata: LightType::Ambient,
        intensity: (0.4, 0.4, 0.4), //white ambient light
    });
    //scene.lights.insert(3, Light {
    //    metadata: LightType::Directional((0.0, 1.0, 1.0)),
    //    intensity: (0.2, 0.2, 0.2),
    //});

    //light intensity normalization
    let mut channel_sums = (0., 0., 0.);
    for light in &scene.lights {
        channel_sums = v_add(&channel_sums, &light.intensity);
    }
    channel_sums = v_rowinvert(&channel_sums);
    for i in 0..scene.lights.len() {
        scene.lights[i].intensity = v_rowmult(&scene.lights[i].intensity, &channel_sums);
    }

    //90 degrees about z+ axis, looking at 0deg wrt. z axis
    //let mut camera_orientation = arr2(&[[1., 0., 0.,], [0., 1., 0.,], [0., 0., 1.,]]);

    let mut viewport_calc_timing = timing::new();
    let mut trace_timing = timing::new();
    
    //draw scene
    for x in (-(CANVAS_WIDTH as i32)/2)..(CANVAS_WIDTH as i32)/2 {
        for y in -(CANVAS_HEIGHT as i32)/2..(CANVAS_HEIGHT as i32)/2 {
            //convert to viewport
            //let viewport_pos: Vec3 = coordinates_canvas_to_viewport(x, y);

            //convert to ndarray type
            let mut viewport_pos = coordinates_canvas_to_viewport(x, y);

            let _viewport_pos = camera.orientation.dot(&vec3_to_arr2(&viewport_pos).t()); //v slow
            // 7 ms per call

            //viewport_calc_timing.start();
            // 2 ms per call -> this is almost the same as the trace ray call!!!!
            arr2_to_vec3(&_viewport_pos.t().to_owned()); //relatively slow
            //viewport_calc_timing.elapse();

            
            //println!("Here");
            //trace to get color of point
            trace_timing.start();
            let color = trace_ray(&scene, &camera.position, &viewport_pos, 1., INFINITY as f32, TRACE_RECURSION_DEPTH);
            trace_timing.elapse();
            //draw on canvas
            let canvas_pos = convert_canvas_coordinates(x, y);

            canvas.set_draw_color(color);
            match canvas.draw_point(canvas_pos) {
                Ok(_) => (),
                Err(str) => println!("err in render: {}", str),
            }
        }   
    }

    println!("finished draw");
    let (view_cnt, view_avg) = viewport_calc_timing.average();
    let (trace_cnt, trace_avg) = trace_timing.average();
    println!("viewport avg calc: {}s over {} entries", view_avg, view_cnt);
    println!("trace calc: {}s over {} entries", trace_avg, trace_cnt);

    //show completed scene
    canvas.present();

    //check for window inputs
    let mut event_pump = sdl.event_pump()?;

    //holding window loop (so it doesn't instantly close on draw),
    //checking for quit event to stop the loop
    'main_loop: loop {
        while let Some(event) = event_pump.poll_event() {
            match event {
                Event::Quit {..} => break 'main_loop,
                _ => (),
            }
        }
    }
    Ok(())
}

fn trace_ray(scene: &Scene, camera_pos: &Vec3, viewport_pos: &Vec3, min_trace: f32, max_trace: f32, recursion_depth: u8) -> Color {
    let (closest_sphere, closest_val) = closest_intersection(scene, camera_pos, viewport_pos, min_trace, max_trace);
    
    if closest_sphere.is_none() { BACKGROUND_COLOR } else {
        //println!("sphere was hit");
        //println!("{}", closest_val);
        let sphere = closest_sphere.unwrap();
        let point = v_add(&camera_pos ,&v_scmult(viewport_pos, closest_val));
        let surface_normal = v_sub(&point, &closest_sphere.unwrap().position);
        let viewpoint_vec = v_scmult(viewport_pos, -1.);

        // setup color vectors for the position
        let mut coerced_sphere_color = color_to_float_channels(&sphere.color);

        //Compute Lighting at point
        let point_intensity = compute_lighting(scene, &point, &surface_normal, &viewpoint_vec, &sphere.specular);
        
        //row mult all channels by the point intensity from the incoming light
        coerced_sphere_color = v_rowmult(&coerced_sphere_color, &point_intensity);

        if recursion_depth <= 0 || sphere.reflective <= 0. {
            return float_channels_to_color(&coerced_sphere_color);
        }

        //calculate reflected ray so we can continue tracing for our reflection
        let reflected_ray = reflect_ray(&v_scmult(viewport_pos, -1.), &surface_normal);
        let reflected_color = trace_ray(scene, &point, &reflected_ray, EPSILON, max_trace, recursion_depth - 1);
        
        //perform the calculation local_color * (1 - r) + reflected_color * r, where r = object reflective constant
        let mut reflected_channels = color_to_float_channels(&reflected_color);
        reflected_channels = v_scmult(&reflected_channels, sphere.reflective); //reflected_color * r
        reflected_channels = v_add(&reflected_channels, &v_scmult(&coerced_sphere_color, 1. - sphere.reflective)); //local * (1-r)
        
        //convert back to color, return
        float_channels_to_color(&reflected_channels)
    }  

}

fn compute_lighting(scene: &Scene, point: &Vec3, surface_normal: &Vec3, viewpoint: &Vec3, specular: &u32) -> Vec3 {
    // current 3-channel intensity at point P
    let mut point_light_intensity: Vec3 = (0., 0., 0.);
    // iterate through all lights, we want to determine how they affect point P w.r.t. the camera
    for light in &scene.lights {
        if light.metadata == LightType::Ambient {
            point_light_intensity = v_add(&point_light_intensity, &light.intensity);
        } else {
            //setup args depending on light type
            let (l_vec, t_max) = match light.metadata {
                LightType::Point(pos) => (v_sub(&pos, point), 1.),
                LightType::Directional(L) => (L, INFINITY as f32),
                LightType::Ambient => panic!("invalid branch of match in compute_lighting"), //covered in former if statement
            };

            //shadow
            let (shadow_casting_sphere, _) = closest_intersection(scene, point, &l_vec, EPSILON, t_max);
            if shadow_casting_sphere.is_some() {
                continue;
            }

            //diffuse
            let n_l_dot = dot(surface_normal, &l_vec);
            if n_l_dot > 0. { 
                let light_reflect_scalar = n_l_dot / (v_len(*surface_normal) * v_len(l_vec));
                point_light_intensity = v_add(&point_light_intensity, &v_scmult(&light.intensity, light_reflect_scalar));
            }

            //specular
            if *specular > 0 {
                let reflection_vec = reflect_ray(&l_vec, surface_normal);
                let r_dot_v = dot(&reflection_vec, viewpoint);

                if r_dot_v > 0. {
                    let quot = v_len(reflection_vec) * v_len(*viewpoint);
                    let point_spec_power = v_scmult(&light.intensity, (r_dot_v/quot).powi(*specular as i32));
                    point_light_intensity = v_add(&point_light_intensity, &point_spec_power);
                }
            }
        }
    }
    point_light_intensity
}

//performs 2 * N * dot(N, R) - R
fn reflect_ray(ray: &Vec3, surface_normal: &Vec3) -> Vec3 {
    let mut inner_product = v_scmult(surface_normal, dot(surface_normal, ray)); //N * dot(N, R)
    inner_product = v_scmult(&inner_product, 2.); //2 * [N * dot(N, R)] (brackets calculated above))
    v_sub(&inner_product, ray) //.. - R
}

//used in trace_ray and compute_lighting
fn closest_intersection<'a>(scene: &'a Scene, camera_pos: &'a Vec3, viewport_pos: &'a Vec3, t_min: f32, t_max: f32) -> (Option<&'a Sphere>, f32) {
    let mut closest_sphere: Option<&Sphere> = None;
    let mut closest_t = INFINITY as f32;

    for i in 0..scene.spheres.len(){
        let cur_sphere = &scene.spheres[i];
        let (t1, t2) = intersect_ray_sphere(camera_pos, viewport_pos, cur_sphere);

        if in_range(t1, t_min as f32, t_max as f32) && t1 < closest_t {
            closest_t = t1;
            closest_sphere = Some(&cur_sphere);
        }

        if in_range(t2, t_min as f32, t_max as f32) && t2 < closest_t {
            closest_t = t2;
            closest_sphere = Some(&cur_sphere);
        }
    }

    (closest_sphere, closest_t)
}

fn scene1(scene: &mut Scene) {
        //define spheres in scene
        scene.spheres.insert(0, Sphere {
            color: Color::RED,
            position: (0., 3.5, -1.),
            radius: 1.,
            specular: 10,
            reflective: 0.9,
        });
        scene.spheres.insert(1, Sphere {
            color: Color::BLUE,
            position: (3., 12.5, 0.),
            radius: 2.,
            specular: 500,
            reflective: 0.5,
        });
        scene.spheres.insert(2, Sphere {
            color: Color::MAGENTA,
            position: (-0.5, 5., 0.),
            radius: 0.5,
            specular: 100,
            reflective: 0.05,
        });
        //should? cast a shadow on the red sphere
        scene.spheres.insert(0, Sphere {
            color: Color::GREEN,
            position: (0.2, 2.0, 0.5),
            radius: 0.2,
            specular: 10,
            reflective: 0.,
        });
    
        //making spheres for each light source
        //todo: debug option that shows spheres for each light source
        scene.spheres.insert(3, Sphere {
            color: Color::WHITE,
            position: ((0.0, 0.0, 0.8)),
            radius: 0.1,
            specular: 10000,
            reflective: 0.,
        });
        scene.spheres.insert(4, Sphere {
            color: Color::WHITE,
            position: ((-7.0, 7.0, 0.0)),
            radius: 0.1,
            specular: 10000,
            reflective: 0.,
        });
    
        
        //define lighting elements
        //note: all light intensities are normalized so that the sum of each channel is 1
        scene.lights.insert(0, Light {
            metadata: LightType::Point((0.0, 0.0, 8.0)),
            intensity: (0.9, 0.9, 0.9), //color channels added, red point light
        });
        scene.lights.insert(1, Light {
            metadata: LightType::Point((-7.0, 7.0, 0.0)),
            intensity: (0.3, 0.3, 0.3),
        });
        scene.lights.insert(2, Light {
            metadata: LightType::Ambient,
            intensity: (0.1, 0.1, 0.1) //white ambient light
        });
}

//solve quadratic for potential hits as line is traced out
fn intersect_ray_sphere(camera_pos: &Vec3, viewport_pos: &Vec3, sphere: &Sphere) -> (f32, f32) {
    //solving ax^2 + bx + c = 0

    let camera_to_sphere_vec: Vec3 = v_sub(camera_pos, &sphere.position);
    let a = dot(viewport_pos, viewport_pos);
    let b = 2. * dot(&camera_to_sphere_vec, viewport_pos);
    let c = dot(&camera_to_sphere_vec, &camera_to_sphere_vec) - (sphere.radius * sphere.radius);

    let mut discriminant = b*b - 4.*a*c;
    if discriminant < 0. { return (INFINITY as f32, INFINITY as f32) }

    discriminant = discriminant.sqrt();
    let quot = 2. * a;

    ( (-b + discriminant) / quot, (-b - discriminant) / quot )
}

fn coordinates_canvas_to_viewport(canvas_x: i32, canvas_y: i32) -> Vec3 { 
    //hard-coded depth from canvas - unit 1 (in y dir)
    (canvas_x as f32 * (VIEWPORT_WIDTH as f32/CANVAS_WIDTH as f32), 1., canvas_y as f32 * (VIEWPORT_HEIGHT as f32/CANVAS_HEIGHT as f32))
}

// from 0,0 centering to array ordering
fn convert_canvas_coordinates(x: i32, y: i32) -> Point { 
    Point::new((CANVAS_WIDTH as i32/2) + x, (CANVAS_HEIGHT as i32/2) - y)
}

// color helpers
#[inline]
fn color_to_float_channels(color: &Color) -> Vec3 {
    (color.r as f32, color.g as f32, color.b as f32)
}

#[inline]
fn float_channels_to_8int(channels: &Vec3) -> (u8, u8, u8) {
    (channels.0.round() as u8, channels.1.round() as u8, channels.2.round() as u8)
}

#[inline]
fn float_channels_to_color(channels: &Vec3) -> Color {
    Color::from(float_channels_to_8int(channels))
}

// f32 helpers
// returns true if v is within [min, max]
#[inline]
fn in_range(v: f32, min: f32, max: f32) -> bool { v >= min && v <= max }

// VEC3 math helpers
// dot product, (u_0 * v_0 + u_1 * v_1 + u_2 * v_2)
#[inline]
fn dot(u: &Vec3, v: &Vec3) -> f32 { u.0 * v.0 + u.1 * v.1 + u.2 * v.2 }
// add v and u (v + u, (u_0 + v_0, u_1 + v_1, u_2 + v_2))
#[inline]
fn v_add(u: &Vec3, v: &Vec3) -> Vec3 { (u.0 + v.0, u.1 + v.1, u.2 + v.2) }
// get vector inverse (-u_0, -u_1, -u_2)
#[inline]
fn v_inv(u: &Vec3) -> Vec3 { (-u.0, -u.1, -u.2) }
// subtract v from u (u - v, e.x. (u_0 - v_0, u_1 - v_1, u_2 - v_2))
fn v_sub(u: &Vec3, v: &Vec3) -> Vec3 {
    let nv = v_inv(v);
    v_add(u, &nv)
}
#[inline]
fn v_scmult(u: &Vec3, b: f32) -> Vec3 { (u.0 * b, u.1 * b, u.2 * b) }
// multiply row-wise, i.e. (v_0 * u_0, v_1 * u_1, v_2 * u_2)
#[inline]
fn v_rowmult(u: &Vec3, v: &Vec3) -> Vec3 { (u.0 * v.0, u.1 * v.1, u.2 * v.2) }
// invert row-wise, (1/v_0, 1/v_1, 1/v_2)
#[inline]
fn v_rowinvert(u: &Vec3) -> Vec3 { (1./u.0, 1./u.1, 1./u.2) }
// get vector length -> this could be amortized by calculating upfront (?) or lazy calculating
#[inline]
fn v_len(u: Vec3) -> f32 { (u.0.powi(2) + u.1.powi(2) + u.2.powi(2)).sqrt() }
// return a unit vector
#[inline]
fn v_norm(u: Vec3) -> Vec3 {
    let len = v_len(u);
    (u.0 / len, u.1 / len, u.2 / len)
}
#[inline]
fn vec3_to_arr2(u: &Vec3) -> Array2<f32> {
    arr2(&[[u.0, u.1, u.2]])
}

#[inline]
fn arr2_to_vec3(u: &Array2<f32>) -> Vec3 {
    (u[[0,0]], u[[0,1]], u[[0,2]])
}

//rot matrix helpers
//b in radians
fn z_rot_mat(b: f32) -> Array2<f32> {
    arr2(&[[b.cos(), -b.sin(), 0.], [b.sin(), -b.cos(), 0.], [0., 0., 1.]])
}