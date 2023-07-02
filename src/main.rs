#![feature(convert_float_to_int)]
pub mod timing;

use lazy_static::lazy_static;
use std::{f32::consts::{PI, FRAC_PI_2}, ops::{Sub, AddAssign}, time::Duration};
use std::convert::FloatToInt;

extern crate nalgebra as na;
use na::{SMatrix, Vector3, Rotation3, Point3, Point2};
use num_traits::{Num, ToPrimitive};
//CHAPTER 3 - updated to incorporate lighting types
use sdl2::{pixels::Color, event::Event, render::Canvas, video::Window};

use crate::timing::TimerTrait;

//TYPEDEFs
type Vec3 = Vector3<f32>;
type Mat3 = SMatrix<f32, 3, 3>;
type Rot3 = Rotation3<f32>;

//STRUCTs
#[derive(Copy, Clone)]
struct PointC {
    point: Point<i32, i32>,
    h: f64,
}

#[derive(Copy, Clone)]
struct Point<T, C> {
    x: T,
    y: C,
}

struct Scene {
    camera: Camera,
    objects: Vec<Object>,
}

struct Camera {
    position: Vec3,
    orientation: Rot3,
}

struct Transform {
    scale: f32,
    rotation: Rot3,
    position: Vec3,
}

struct Object {
    model: &'static Model,
    transform: Transform,
    //more...
}

//TRAITs & IMPLs
trait Reverse<T, C> {
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

struct Model {
    vertices: Vec<Vec3>,
    triangles: Vec<(usize, usize, usize, Color)>,
}

//instantiate models
lazy_static! {
    static ref MODEL_CUBE: Model = Model {
        vertices: vec![
            Vec3::new(1., 1., 1.),
            Vec3::new(-1., 1., 1.),
            Vec3::new(-1., -1., 1.),
            Vec3::new(1., -1., 1.),
            Vec3::new(1., 1., -1.),
            Vec3::new(-1., 1., -1.),
            Vec3::new(-1., -1., -1.),
            Vec3::new(1., -1., -1.),
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
}

//CONSTs
const BACKGROUND_COLOR: Color = Color::GRAY;//Color::WHITE;
const CANVAS_WIDTH: u32 = 800;
const CANVAS_HEIGHT: u32 = 800;

const VIEWPORT_WIDTH: f32 = 1.0;
const VIEWPORT_HEIGHT: f32 = 1.0;
const VIEWPORT_Y_DIST: f32 = 1.0;
pub fn main() -> Result<(), String> {
    let sdl = sdl2::init()?;
    let window = sdl.video()?.window("Rasterizer", CANVAS_WIDTH, CANVAS_HEIGHT)
        .position_centered()
        .opengl()
        .build()
        .map_err(|e| e.to_string())?;

    let mut canvas = window.into_canvas().build().map_err(|e| e.to_string())?;
    let mut scene = init_scene();

    scene.objects.push(Object {
        model: &MODEL_CUBE,
        transform: Transform {
            position: Vec3::new(1.4, 10.0, 0.0),
            rotation: Rot3::from_axis_angle(&Vec3::z_axis(), 0.),
            scale: 1.,
        },
    });

    scene.objects.push(Object { 
        model: &MODEL_CUBE,
        transform: Transform{
            position: Vec3::new(-1.0, 9.0, 1.1),
            rotation: Rot3::from_axis_angle(&Vec3::y_axis(), 1.2),
            scale: 0.9,
        },
    });

    let mut event_pump = sdl.event_pump()?;
    let mut running = true;
    let mut trans_z_pos: f32 = 0.0;

    while running {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit {..} => running = false,
                _ => (),
            }
        }
        trans_z_pos += 0.01;
        
        scene.objects[0].transform.position = Vec3::new(1.5, 10., f32::cos(trans_z_pos));
        scene.objects[0].transform.rotation = Rot3::from_axis_angle(&Vec3::x_axis(), trans_z_pos*2.);
        scene.objects[1].transform.rotation = Rot3::from_axis_angle(&Vec3::z_axis(), trans_z_pos);
        scene.objects[1].transform.position = Vec3::new(-1. + f32::cos(trans_z_pos), 8. + 2.*f32::sin(trans_z_pos), 0.);
        render_scene(&mut canvas, &scene);
        std::thread::sleep(Duration::from_millis(1));
    }
    Ok(())
}

    //canvas.set_draw_color(BACKGROUND_COLOR);
    //canvas.clear(); //set bg to be black

    //for instance in scene.objects {
    //    render_object(&mut canvas, &instance);
    //}

    /*  //draw scene
    for x in (-(CANVAS_WIDTH as i32)/2)..(CANVAS_WIDTH as i32)/2 {
        for y in -(CANVAS_HEIGHT as i32)/2..(CANVAS_HEIGHT as i32)/2 {

            let canvas_pos = x, y;


            canvas.set_draw_color(color);
            match canvas.draw_point(canvas_pos) {
                Ok(_) => (),
                Err(str) => println!("err in render: {}", str),
            }
        }   
    }*/

    //draw_line(&mut canvas, Point{x: 100, y: 100}, Point{x: -100, y: -100}, Color::BLACK);
    //draw_line(&mut canvas, Point{x: -100, y: 100}, Point{x: 100, y: -100}, Color::BLACK);

    //draw_line(&mut canvas, Point{x: -399, y: -299}, Point{x: -399, y: 299}, Color::RED);
    //draw_line(&mut canvas, Point{x: 399, y: -299}, Point{x: 399, y: 299}, Color::GREEN);
    //draw_line(&mut canvas, Point{x: -399, y: -299}, Point{x: 399, y: -299}, Color::MAGENTA);
    //draw_line(&mut canvas, Point{x: -399, y: 299}, Point{x: 399, y: 299}, Color::BLUE);

    //draw_triangle(&mut canvas, Point{x: -100, y: 0}, Point{x: 0, y: 100}, Point{x: 100, y: 0}, Color::RED);
    /*
    draw_triangle_shaded(&mut canvas, 
        PointC {point: Point{x: -200, y: -50}, h: 1.0},
        PointC {point: Point{x: -100, y: -50}, h: 0.2},
        PointC {point: Point{x: -1, y: -200}, h: 0.0},
        Color::YELLOW
    );

    draw_triangle_shaded(&mut canvas, 
        PointC {point: project_vertex(Vec3::new(2.0, 5.5, 1.0)), h: 0.2},
        PointC {point: project_vertex(Vec3::new(1.0, 5.0, 1.5)), h: 0.4},
        PointC {point: project_vertex(Vec3::new(0.0, 5.5, 1.0)), h: 0.8},
        Color::MAGENTA
    );*/
    //draw_triangle_shaded(&mut canvas, 
    //    PointC {
    //        point: project_vertex(Point{x: 100, y: -230}), h: 0.2},
    //    PointC {point: Point{x: 168, y: -20}, h: 0.4},
    //    PointC {point: Point{x: 201, y: -200}, h: 0.8},
    //    Color::MAGENTA
    //);

    //project_cube(&mut canvas, Vec3::new(-1.5, 5.5, 0.0), 1.0);
    //project_cube(&mut canvas, Vec3::new(1.0, 4.5, 0.0), 0.2);

    //render_wireframe_cube(&mut canvas, Vec3::new(1.0, 3.4, 0.0), 0.5);

    //show completed scene
    //canvas.present();



fn render_scene(canvas: &mut Canvas<Window>, scene: &Scene) {
    
    canvas.set_draw_color(BACKGROUND_COLOR);
    canvas.clear(); //set bg to be black

    for instance in scene.objects.as_slice() {
        render_object(canvas, instance);
    }

    canvas.present();
}

fn draw_line(canvas: &mut Canvas<Window>, mut p1: Point<i32, i32>, mut p2: Point<i32, i32>, color: Color) {    
    if (p2.x - p1.x).abs() > (p2.y - p1.y).abs() { //horizontal-ish
        if p1.x > p2.x {
            swap(&mut p1, &mut p2);
        }
        let interps_y = lerp_i32(&p1, &p2);
        for x in p1.x..p2.x+1 {
            put_pixel(canvas, x, interps_y[(x - p1.x) as usize], color)
        }

    } else { //vertical-ish
        if p1.y > p2.y {
            swap(&mut p1, &mut p2);
        }
        let interps_x = lerp_i32(&p1.reverse(), &p2.reverse());
        for y in p1.y..p2.y+1 {
            put_pixel(canvas, interps_x[(y - p1.y) as usize], y, color)
        }
    }

}

fn draw_triangle(canvas: &mut Canvas<Window>, mut v0: Point<i32, i32>, mut v1: Point<i32, i32>, mut v2: Point<i32, i32>, color: Color) {

    // ordering
    //v1: left-most
    //[y0, y2] (v1 has lowest y)

    if v2.y < v1.y { 
        swap(&mut v2, &mut v1); //assert y3 > y2
    }
    if v2.y < v0.y {
        swap(&mut v2, &mut v0) //assert y3 > y1
    }
    if v1.y < v0.y {
        swap(&mut v1, &mut v0);
    }

    // assert v3.y > v2.y > v1.y
    assert!((v2.y >= v1.y) && (v1.y >= v0.y), "vertices were not ordered properly");

    let mut x01 = lerp_i32(&v0.reverse(), &v1.reverse());
    let x12 = lerp_i32(&v1.reverse(), &v2.reverse());
    let x02 = lerp_i32(&v0.reverse(), &v2.reverse()); //tall
    x01 = x01[0..x01.len() - 1].to_vec(); //remove last for concatenating
    let x012 = [x01, x12].concat();

    let xleft: Vec<i32>;
    let xright: Vec<i32>;

    let point = x02.len() / 2;
    if x02[point] < x012[point]  {
        xleft = x02;
        xright = x012;
    } else {
        xleft = x012;
        xright = x02;
    }

    for y in v0.y..v2.y+1 {
        let cur_y = (y - v0.y) as usize;
        for x in xleft[cur_y]..xright[cur_y]+1 {
            put_pixel(canvas, x, y, color);
        }
    }

    draw_line(canvas, v0, v1, Color::GREEN); //leftmost top-bottom line
    draw_line(canvas, v1, v2, Color::BLUE); //rightmost top-bottom line
    draw_line(canvas, v2, v0, Color::RED);
    


    //for y in min_y..max_y+1 {
    //    draw_line(canvas, Point::from());
    //}
}


// Point3: x,y,h, s.t. h corresponds to color intensity at vertex given vertex vi
fn draw_triangle_shaded(canvas: &mut Canvas<Window>, mut v0: PointC, mut v1: PointC, mut v2: PointC, color: Color) {

    // ordering
    //v1: left-most
    //[y0, y2] (v1 has lowest y)

    if v2.point.y < v1.point.y { 
        swap(&mut v2, &mut v1); //assert y3 > y2
    }
    if v2.point.y < v0.point.y {
        swap(&mut v2, &mut v0) //assert y3 > y1
    }
    if v1.point.y < v0.point.y {
        swap(&mut v1, &mut v0);
    }

    // assert v3.y > v2.y > v1.y
    assert!((v2.point.y >= v1.point.y) && (v1.point.y >= v0.point.y), "vertices were not ordered properly");

    let v0h_loc = Point{x: v0.point.y, y: v0.h};
    let v1h_loc = Point{x: v1.point.y, y: v1.h};
    let v2h_loc = Point{x: v2.point.y, y: v2.h};

    let mut x01 = lerp_i32(&v0.point.reverse(), &v1.point.reverse());
    let mut h01 = lerp(&v0h_loc, &v1h_loc);

    let x12 = lerp_i32(&v1.point.reverse(), &v2.point.reverse());
    let h12 = lerp(&v1h_loc, &v2h_loc);

    let x02 = lerp_i32(&v0.point.reverse(), &v2.point.reverse()); //tall
    let h02 = lerp(&v0h_loc, &v2h_loc);
    x01 = x01[0..x01.len() - 1].to_vec(); //remove last for concatenating
    h01 = h01[0..h01.len() - 1].to_vec();
    let x012 = [x01, x12].concat();
    let h012 = [h01, h12].concat();

    let xleft: Vec<i32>;
    let xright: Vec<i32>;

    let hleft: Vec<f64>;
    let hright: Vec<f64>;

    let point = x02.len() / 2;
    if x02[point] < x012[point]  {
        xleft = x02;
        xright = x012;
        hleft = h02;
        hright = h012;
    } else {
        xleft = x012;
        xright = x02;
        hleft = h012;
        hright = h02;
    }

    for y in v0.point.y..v2.point.y+1 {
        let cur_y = (y - v0.point.y) as usize;

        let xl = xleft[cur_y];
        let xr = xright[cur_y];

        let h_segment = lerp(&Point{x: xl, y: hleft[cur_y]}, &Point{x: xr, y: hright[cur_y]});        
        for x in xl..xr+1 {
            let shade = color_to_float_channels(&color) * h_segment[(x - xl) as usize] as f32;
            put_pixel(canvas, x, y, float_channels_to_color(&shade));
        }
    }

    //draw_line(canvas, v0, v1, Color::GREEN); //leftmost top-bottom line
    //draw_line(canvas, v1, v2, Color::BLUE); //rightmost top-bottom line
    //draw_line(canvas, v2, v0, Color::RED);
    


    //for y in min_y..max_y+1 {
    //    draw_line(canvas, Point::from());
    //}
}

fn draw_wireframe_triangle (canvas: &mut Canvas<Window>, v0: Point<i32, i32>, v1: Point<i32, i32>, v2: Point<i32, i32>, color: Color) {
    draw_line(canvas, v0, v1, color); //leftmost top-bottom line
    draw_line(canvas, v1, v2, color); //rightmost top-bottom line
    draw_line(canvas, v2, v0, color);
}

#[inline]
fn swap<T>(s1: &mut T, s2: &mut T) 
where
    T: Copy,
{
    let temp = *s1;
    *s1 = *s2;
    *s2 = temp;
}

//LERP values from point_i to the final position point_f
//notice: the first point value is the 'independent' variable, the second is the 'dependent' (ex, x -> y, ...)
//this gives us a 1 to multiple relationship
fn lerp<T, C>(point_i: &Point<T, C>, point_f: &Point<T, C>) -> Vec<C>
where
    T: Num + PartialOrd + Clone + ToPrimitive + Into<C> + Copy,
    C: Num + AddAssign + From<f32> + Copy,
{
    if point_i.x == point_f.x { // i_0 == i_1 
        return vec!(point_i.y)
    }
    let mut values = Vec::new(); //set of all final dependent variables per int value of i [i0.. i1]..
    // we could optimize to not use FP division for our line drawing

    let slope: C = (point_f.y - point_i.y) / (point_f.x - point_i.x).into();//.to_f32().expect("Improper lerp slope."); //purposely kept as float - handle rounding at addition
    let mut d: C = point_i.y; //.to_f32().expect("could not coerce point dependent var into float");

    for _ in num_iter::range_inclusive(point_i.x, point_f.x) { //we want an inclusive range
        values.push(d.into());
        d += slope;
    }
    values
}

//LERP for integer dependent-variable.
//due to lossy nature of fp->int transform, it is difficult to handle int transforms in
//Rust (without using unsafe blocks, etc) so a separate function has been made
//also makes handling rounding easier
fn lerp_i32(point_i: &Point<i32, i32>, point_f: &Point<i32, i32>) -> Vec<i32> {
    if point_i.x == point_f.x { // i_0 == i_1 
        return vec!(point_i.y)
    }
    let mut values = Vec::new(); //set of all final dependent variables per int value of i [i0.. i1]..
    // we could optimize to not use FP division for our line drawing

    let slope: f32 = (point_f.y - point_i.y) as f32 / (point_f.x - point_i.x) as f32;
    let mut d: f32 = point_i.y.to_f32().expect("could not coerce point dependent var into float");

    for _ in num_iter::range_inclusive(point_i.x, point_f.x) { //we want an inclusive range
        values.push(d.round() as i32);
        d += slope;
    }
    values
}

#[inline]
fn put_pixel(canvas: &mut Canvas<Window>, x: i32, y: i32, color: Color) {
    canvas.set_draw_color(color);
    match canvas.draw_point(sdl2::rect::Point::from(canvas_to_zero_ind(x, y))) {
        Ok(_) => {},
        Err(_) => println!("Error in put pixel for ({},{})", x, y),
    };
}

//we make a set of vertices with indexes 0..n, and triangles, which point to the vertices
// i.e.
//vertices: 0=(1.0, 2.0, 4.0), 1, 2, ...
//triangles: 0=(v1, v3, v2, color:red)...
fn render_object(canvas: &mut Canvas<Window>, object: &Object) {
    let mut projected_vertices = Vec::new();
    for v in object.model.vertices.as_slice() {
        let t_vertex = apply_transform(*v, &object.transform);
        projected_vertices.push(project_vertex(t_vertex));
    }
    for (v0, v1, v2, color) in object.model.triangles.as_slice() {
        draw_wireframe_triangle(canvas, projected_vertices[*v0], projected_vertices[*v1], projected_vertices[*v2], *color);
    }

}

fn render_wireframe_cube(canvas: &mut Canvas<Window>, center: Vec3, size: f32) {
    let dist = size/2.;

    let mut vertices = Vec::new();
    let mut triangles = Vec::new();

    let v_front_face = center - Vec3::new(0., dist, 0.);
    let v_back_face = center + Vec3::new(0., dist, 0.);

    let v_fr_bl = v_front_face + Vec3::new(-dist, 0., -dist);
    let v_fr_br = v_front_face + Vec3::new(dist, 0., -dist);
    let v_fr_tl = v_front_face + Vec3::new(-dist, 0., dist);
    let v_fr_tr = v_front_face + Vec3::new(dist, 0., dist);

    let v_ba_bl = v_back_face + Vec3::new(-dist, 0., -dist);
    let v_ba_br = v_back_face + Vec3::new(dist, 0., -dist);
    let v_ba_tl = v_back_face + Vec3::new(-dist, 0., dist);
    let v_ba_tr = v_back_face + Vec3::new(dist, 0., dist);

    //front vertices
    vertices.push(v_fr_bl); //0 = front BL
    vertices.push(v_fr_br); //1 = front BR
    vertices.push(v_fr_tl); //2 = front TL
    vertices.push(v_fr_tr); //3 = front TR

    vertices.push(v_ba_bl); //4 = back BL
    vertices.push(v_ba_br); //5 = back BR
    vertices.push(v_ba_tl); //6 = back TL
    vertices.push(v_ba_tr); //7 = back TR

    //BACK FACE
    triangles.push((0, 1, 4, Color::BLUE)); //bottom-left triangle BACK
    triangles.push((1, 4, 5, Color::BLUE)); //top-right triangle BACK

    //BOTTOM FACE
    triangles.push((0, 1, 4, Color::CYAN));
    triangles.push((1, 4, 5, Color::CYAN));

    //TOP FACE
    triangles.push((2, 3, 6, Color::YELLOW));
    triangles.push((3, 6, 7, Color::YELLOW));

    //LEFT FACE
    triangles.push((0, 4, 6, Color::GREEN)); //bottom-left triangle LEFT
    triangles.push((0, 2, 6, Color::GREEN)); //top-right triangle LEFT

    //RIGHT FACE
    triangles.push((1, 5, 7, Color::MAGENTA)); //bottom-right triangle RIGHT
    triangles.push((1, 3, 7, Color::MAGENTA)); //top-left triangle RIGHT

    //FRONT FACE
    triangles.push((0, 1, 2, Color::RED)); //bottom-left triangle FRONT
    triangles.push((1, 2, 3, Color::RED)); //top-right triangle FRONT


    //render_object(canvas, vertices, triangles)
}

/// draw cube "facing" y-axis
fn project_cube(canvas: &mut Canvas<Window>, d: Vec3, size: f32) {
    let dist = size/2.;

    let v_front_face = d - Vec3::new(0., dist, 0.);
    let v_back_face = d + Vec3::new(0., dist, 0.);

    let v_fr_bl = v_front_face + Vec3::new(-dist, 0., -dist);
    let v_fr_br = v_front_face + Vec3::new(dist, 0., -dist);
    let v_fr_tl = v_front_face + Vec3::new(-dist, 0., dist);
    let v_fr_tr = v_front_face + Vec3::new(dist, 0., dist);

    let v_ba_bl = v_back_face + Vec3::new(-dist, 0., -dist);
    let v_ba_br = v_back_face + Vec3::new(dist, 0., -dist);
    let v_ba_tl = v_back_face + Vec3::new(-dist, 0., dist);
    let v_ba_tr = v_back_face + Vec3::new(dist, 0., dist);

    //BACK FACE
    draw_line(canvas, project_vertex(v_ba_bl), project_vertex(v_ba_br), Color::RED);
    draw_line(canvas, project_vertex(v_ba_br), project_vertex(v_ba_tr), Color::RED);
    draw_line(canvas, project_vertex(v_ba_tr), project_vertex(v_ba_tl), Color::RED);
    draw_line(canvas, project_vertex(v_ba_tl), project_vertex(v_ba_bl), Color::RED);

    //CONNECTING LINES
    draw_line(canvas, project_vertex(v_ba_bl), project_vertex(v_fr_bl), Color::GREEN);
    draw_line(canvas, project_vertex(v_ba_br), project_vertex(v_fr_br), Color::GREEN);
    draw_line(canvas, project_vertex(v_ba_tr), project_vertex(v_fr_tr), Color::GREEN);
    draw_line(canvas, project_vertex(v_ba_tl), project_vertex(v_fr_tl), Color::GREEN);    

    //FRONT FACE
    draw_line(canvas, project_vertex(v_fr_bl), project_vertex(v_fr_br), Color::BLUE);
    draw_line(canvas, project_vertex(v_fr_br), project_vertex(v_fr_tr), Color::BLUE);
    draw_line(canvas, project_vertex(v_fr_tr), project_vertex(v_fr_tl), Color::BLUE);
    draw_line(canvas, project_vertex(v_fr_tl), project_vertex(v_fr_bl), Color::BLUE);
}

fn init_scene() -> Scene {
    Scene {
        camera: Camera { 
            position: Vec3::zeros(), 
            orientation: Rot3::from_axis_angle(&Vec3::z_axis(), 0.) 
        },
        objects: Vec::new(),
    }
}

#[inline]
fn apply_transform(mut vertex: Vec3, transform: &Transform) -> Vec3 {
    vertex *= transform.scale;
    vertex = transform.rotation * vertex;
    vertex += transform.position;
    vertex
}

//CANVAS/VIEWPORT helpers

/// convert viewport (x,y) 'real' values into canvas pixel values
#[inline]
fn viewport_to_canvas(x: f32, y: f32) -> Point<i32, i32> {
    Point {
        x: (x * (CANVAS_WIDTH as f32/VIEWPORT_WIDTH)).round() as i32,
        y: (y * (CANVAS_HEIGHT as f32/VIEWPORT_HEIGHT)).round() as i32,
    }
}

#[inline]
fn project_vertex(vertex: Vec3) -> Point<i32, i32> {
    viewport_to_canvas(vertex.x * VIEWPORT_Y_DIST / vertex.y, vertex.z * VIEWPORT_Y_DIST / vertex.y) //make use of similar triangles to project vertices
}

/// convert from 0-centered 'indexing' to memory-layout (0-index layout)
#[inline]
fn canvas_to_zero_ind(x: i32, y: i32) -> (i32, i32) {
    (((CANVAS_WIDTH as f32/2. + x as f32).round() as i32), ((CANVAS_HEIGHT as f32/2. - y as f32).round() as i32))
}

// color helpers
#[inline]
fn color_to_float_channels(color: &Color) -> Vec3 {
    Vec3::new(color.r as f32, color.g as f32, color.b as f32)
}

#[inline]
fn float_channels_to_8int(channels: &Vec3) -> (u8, u8, u8) {
    (channels[0].round() as u8, channels[1].round() as u8, channels[2].round() as u8)
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
#[inline]
fn v_rowmult(u: &Vec3, v: &Vec3) -> Vec3 { Vec3::new(u[0] * v[0], u[1] * v[1], u[2] * v[2]) }
// invert row-wise, (1/v_0, 1/v_1, 1/v_2)
#[inline]
fn v_rowinvert(u: &Vec3) -> Vec3 { Vec3::new(1./u[0], 1./u[1], 1./u[2]) }
// get vector length -> this could be amortized by calculating upfront (?) or lazy calculating