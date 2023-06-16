//CHAPTER 3 - updated to incorporate lighting types
use sdl2::{pixels::Color, rect::Point, event::Event};

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
    specular: u32, // this should be a general object/surface value
}

struct Light {
    metadata: LightType,
    intensity: Vec3,
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

//this is renderer V1, we use hard-coded camera position
const CAMERA_POSITION: Vec3 = (0., 0., 0.); //origin of world
const BACKGROUND_COLOR: Color = Color::RGB(0, 0, 0);

fn main() {
    let sdl = sdl2::init().unwrap();
    let window = sdl.video().unwrap().window("Renderer", CANVAS_WIDTH, CANVAS_HEIGHT).build().unwrap();
    let mut canvas = window.into_canvas().present_vsync().build().unwrap();

    canvas.set_draw_color(BACKGROUND_COLOR);
    canvas.clear(); //set bg to be black

    //canvas.set_draw_color(Color::RGB(255,0,0));
    //canvas.draw_line(Point::new(0,0),  Point::new(100,100)).unwrap();

    //define scene
    let mut scene = Scene {
        spheres: Vec::new(),
        lights: Vec::new(),
    };

    //define spheres in scene
    scene.spheres.insert(0, Sphere {
        color: Color::RED,
        position: (0., 3.5, -1.),
        radius: 1.,
        specular: 10,
    });
    scene.spheres.insert(1, Sphere {
        color: Color::BLUE,
        position: (3., 12.5, 0.),
        radius: 2.,
        specular: 500,
    });
    scene.spheres.insert(2, Sphere {
        color: Color::MAGENTA,
        position: (-0.5, 5., 0.),
        radius: 0.5,
        specular: 100,
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

    //light intensity normalization
    let mut channel_sums = (0., 0., 0.);
    for light in &scene.lights {
        channel_sums = v_add(&channel_sums, &light.intensity);
    }
    channel_sums = v_rowinvert(&channel_sums);
    for i in 0..scene.lights.len() {
        scene.lights[i].intensity = v_rowmult(&scene.lights[i].intensity, &channel_sums);
    }
    
    //draw scene
    for x in (-(CANVAS_WIDTH as i32)/2)..(CANVAS_WIDTH as i32)/2 {
        for y in -(CANVAS_HEIGHT as i32)/2..(CANVAS_HEIGHT as i32)/2 {
            //convert to viewport
            let viewport_pos: Vec3 = CoordinatesCanvasToViewport(x, y);

            //trace to get color of point
            let color = TraceRay(&scene, &CAMERA_POSITION, &viewport_pos, 1, INFINITY);
            
            //draw on canvas
            let canvas_pos = ConvertCanvasCoordinates(x, y);

            canvas.set_draw_color(color);
            match canvas.draw_point(canvas_pos) {
                Ok(_) => (),
                Err(str) => println!("err in render: {}", str),
            }
        }   
    }

    //show completed scene
    canvas.present();
    
    //check for window inputs
    let mut event_pump = sdl.event_pump().unwrap();

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
}

fn TraceRay(scene: &Scene, camera_pos: &Vec3, viewport_pos: &Vec3, min_trace: i32, max_trace: i32) -> Color {
    let mut closest_val = INFINITY as f32;
    let mut closest_sphere: Option<&Sphere> = None;

    for i in 0..scene.spheres.len(){
        let cur_sphere = &scene.spheres[i];
        let (t1, t2) = IntersectRaySphere(camera_pos, viewport_pos, cur_sphere);

        if in_range(t1, min_trace as f32, max_trace as f32) && t1 < closest_val {
            closest_val = t1;
            closest_sphere = Some(&cur_sphere);
        }

        if in_range(t2, min_trace as f32, max_trace as f32) && t2 < closest_val {
            closest_val = t2;
            closest_sphere = Some(&cur_sphere);
        }
    }
    
    if closest_sphere.is_none() { BACKGROUND_COLOR } else {
        //println!("sphere was hit");
        //println!("{}", closest_val);
        let sphere = closest_sphere.unwrap();
        let point = v_add(&camera_pos ,&v_scmult(viewport_pos, closest_val));
        let surface_normal = v_sub(&point, &closest_sphere.unwrap().position);
        let viewpoint_vec = v_scmult(viewport_pos, -1.);

        // setup color vectors for the position
        let mut sphere_color = sphere.color.rgb();
        let mut coerced_sphere_color = (sphere_color.0 as f32, sphere_color.1 as f32, sphere_color.2 as f32);

        //Compute Lighting at point
        let point_intensity = ComputeLighting(scene, &point, &surface_normal, &viewpoint_vec, &sphere.specular);

        coerced_sphere_color = (coerced_sphere_color.0 * point_intensity.0, coerced_sphere_color.1 * point_intensity.1, coerced_sphere_color.2 * point_intensity.2);
        sphere_color = (coerced_sphere_color.0.round() as u8, coerced_sphere_color.1.round() as u8, coerced_sphere_color.2.round() as u8);
        Color::from(sphere_color)
    }

}

fn ComputeLighting(scene: &Scene, point: &Vec3, surface_normal: &Vec3, viewpoint: &Vec3, specular: &u32) -> Vec3 {
    // current 3-channel intensity at point P
    let mut point_light_intensity: Vec3 = (0., 0., 0.);
    // iterate through all lights, we want to determine how they affect point P w.r.t. the camera
    for light in &scene.lights {
        if light.metadata == LightType::Ambient {
            point_light_intensity = v_add(&point_light_intensity, &light.intensity);
        } else {
            //diffuse
            let l_vec: Vec3 = match light.metadata {
                LightType::Point(pos) => v_sub(&pos, point),
                LightType::Directional(L) => L,
                LightType::Ambient => (0., 0., 0.), //covered in former if statement
            };
            let n_l_dot = dot(surface_normal, &l_vec);
            if n_l_dot > 0. { 
                let light_reflect_scalar = n_l_dot / (v_len(*surface_normal) * v_len(l_vec));
                point_light_intensity = v_add(&point_light_intensity, &v_scmult(&light.intensity, light_reflect_scalar));
            }

            //specular
            if *specular > 0 {
                //println!("spec");
                let mut reflection_vec = v_scmult(surface_normal, 2. * dot(surface_normal, &l_vec)); // R = 2*N*dot(N, L)
                reflection_vec = v_sub(&reflection_vec, &l_vec); //... - L
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

//solve quadratic for potential hits as line is traced out
fn IntersectRaySphere(camera_pos: &Vec3, viewport_pos: &Vec3, sphere: &Sphere) -> (f32, f32) {
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

fn CoordinatesCanvasToViewport(canvas_x: i32, canvas_y: i32) -> Vec3 { 
    //hard-coded depth from canvas - unit 1 (in y dir)
    (canvas_x as f32 * (VIEWPORT_WIDTH as f32/CANVAS_WIDTH as f32), 1., canvas_y as f32 * (VIEWPORT_HEIGHT as f32/CANVAS_HEIGHT as f32))
}

// from 0,0 centering to array ordering
fn ConvertCanvasCoordinates(x: i32, y: i32) -> Point { 
    Point::new((CANVAS_WIDTH as i32/2) + x, (CANVAS_HEIGHT as i32/2) - y)
}

// f32 helpers
// returns true if v is within [min, max]
fn in_range(v: f32, min: f32, max: f32) -> bool { v >= min && v <= max }

// VEC3 math helpers
// dot product, (u_0 * v_0 + u_1 * v_1 + u_2 * v_2)
fn dot(u: &Vec3, v: &Vec3) -> f32 { u.0 * v.0 + u.1 * v.1 + u.2 * v.2 }
// add v and u (v + u, (u_0 + v_0, u_1 + v_1, u_2 + v_2))
fn v_add(u: &Vec3, v: &Vec3) -> Vec3 { (u.0 + v.0, u.1 + v.1, u.2 + v.2) }
// get vector inverse (-u_0, -u_1, -u_2)
fn v_inv(u: &Vec3) -> Vec3 { (-u.0, -u.1, -u.2) }
// subtract v from u (u - v, e.x. (u_0 - v_0, u_1 - v_1, u_2 - v_2))
fn v_sub(u: &Vec3, v: &Vec3) -> Vec3 {
    let nv = v_inv(v);
    v_add(u, &nv)
}
fn v_scmult(u: &Vec3, b: f32) -> Vec3 { (u.0 * b, u.1 * b, u.2 * b) }
// multiply row-wise, i.e. (v_0 * u_0, v_1 * u_1, v_2 * u_2)
fn v_rowmult(u: &Vec3, v: &Vec3) -> Vec3 { (u.0 * v.0, u.1 * v.1, u.2 * v.2) }
// invert row-wise, (1/v_0, 1/v_1, 1/v_2)
fn v_rowinvert(u: &Vec3) -> Vec3 { (1./u.0, 1./u.1, 1./u.2) }
// get vector length -> this could be amortized by calculating upfront (?) or lazy calculating
fn v_len(u: Vec3) -> f32 { (u.0.powi(2) + u.1.powi(2) + u.2.powi(2)).sqrt() }
// return a unit vector
fn v_norm(u: Vec3) -> Vec3 {
    let len = v_len(u);
    (u.0 / len, u.1 / len, u.2 / len)
}