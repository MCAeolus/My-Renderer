use std::ops::AddAssign;
use num_traits::{ToPrimitive, Num};
use sdl2::{render::Canvas, video::Window, pixels::Color};
use crate::{structs::{Point, PointC, Reverse, Vec3, DepthPoint, Light, LightType, RenderTriangle, Scene}, CANVAS_WIDTH, CANVAS_HEIGHT, VIEWPORT_Z_DIST};

pub fn draw_line(canvas: &mut Canvas<Window>, z_buffer: &mut Vec<f32>, mut p1: DepthPoint, mut p2: DepthPoint, color: Color) {
    clamp_point(&mut p1, (-(CANVAS_WIDTH as i32)/2, CANVAS_WIDTH as i32/2 - 1), (-(CANVAS_HEIGHT as i32)/2, CANVAS_HEIGHT as i32/2 - 1));
    clamp_point(&mut p2, (-(CANVAS_WIDTH as i32/2), CANVAS_WIDTH as i32/2 - 1), (-(CANVAS_HEIGHT as i32)/2, CANVAS_HEIGHT as i32/2 - 1));

    if (p2.point.x - p1.point.x).abs() > (p2.point.y - p1.point.y).abs() { //horizontal-ish
        if p1.point.x > p2.point.x {
            swap(&mut p1, &mut p2);
        }
        let interps_y: Vec<i32> = lerp_i32(&p1.point, &p2.point);
        let z_segment = lerp(&Point{x: p1.point.x, y: 1./p1.z as f64}, &Point{x: p2.point.x, y: 1./p2.z as f64});
        //dbg!(p1.x, p2.x);
        for x in p1.point.x..p2.point.x+1 {
            let ind = (x - p1.point.x) as usize;
            let y = interps_y[ind];
            let z = z_segment[ind] as f32;
            let (_x, _y) = canvas_to_zero_ind(x, y);
            let z_ind = (_y * CANVAS_WIDTH as i32 + _x) as usize;
            if z > z_buffer[z_ind] {
                put_pixel(canvas, x, y, color);
                z_buffer[z_ind] = z;
            }
        }

    } else { //vertical-ish
        if p1.point.y > p2.point.y {
            swap(&mut p1, &mut p2);
        }
        let interps_x = lerp_i32(&p1.point.reverse(), &p2.point.reverse());
        let z_segment = lerp(&Point{x: p1.point.y, y: 1./p1.z as f64}, &Point{x: p2.point.y, y: 1./p2.z as f64});
        //dbg!(p1.y, p2.y);
        for y in p1.point.y..p2.point.y+1 {
            let ind = (y - p1.point.y) as usize;
            let x = interps_x[ind];
            let z = z_segment[ind] as f32;

            let (_x, _y) = canvas_to_zero_ind(x, y);
            let z_ind = (_y * CANVAS_WIDTH as i32 + _x as i32) as usize;
            if z > z_buffer[z_ind] {
                put_pixel(canvas, x, y, color);
                z_buffer[z_ind] = z;
            }
        }
    }

}

pub fn compute_illumination(lights: &Vec<Light>, viewpoint: &Vec3, point: &Vec3, surface_normal: &Vec3, specular: &u32) -> Vec3 {
    // current 3-channel intensity at point P
    let mut point_light_intensity: Vec3 = Vec3::zeros();
    // iterate through all lights, we want to determine how they affect point P w.r.t. the camera
    for light in lights {
        if light.metadata == LightType::Ambient {
            point_light_intensity = point_light_intensity + light.intensity;
            //point_light_intensity = v_add(&point_light_intensity, &light.intensity);
        } else {
            //diffuse
            let l_vec: Vec3 = match light.metadata {
                LightType::Point(pos) => pos - point,
                LightType::Directional(L) => L,
                LightType::Ambient => Vec3::zeros(), //covered in former if statement
            };
            let normal_light_dot = surface_normal.dot(&l_vec);
            //let n_l_dot = dot(surface_normal, &l_vec);
            if normal_light_dot > 0. { 
                let light_reflect_scalar = normal_light_dot / (surface_normal.norm() * l_vec.norm());
                point_light_intensity = point_light_intensity + (light.intensity * light_reflect_scalar);
            }

            //specular
            if *specular > 0 {
                //println!("spec");
                let mut reflection_vec = surface_normal * 2. * surface_normal.dot(&l_vec); // R = 2*N*dot(N, L)
                reflection_vec = reflection_vec - l_vec; //... - L
                let r_dot_v = reflection_vec.dot(&viewpoint);

                if r_dot_v > 0. {
                    let quot = reflection_vec.norm() * viewpoint.norm();
                    let point_spec_power = light.intensity * (r_dot_v/quot).powi(*specular as i32);
                    point_light_intensity = point_light_intensity + point_spec_power;
                }
            }
        }
    }
    point_light_intensity
}

//TODO: extending lerp so we can reduce the size of this function would be smart
pub fn draw_triangle(scene: &Scene, canvas: &mut Canvas<Window>, z_buffer: &mut Vec<f32>, triangle: &RenderTriangle) {
    // ordering
    //v1: left-most
    //[y0, y2] (v1 has lowest y)
    let (mut v0, mut v1, mut v2) = triangle.points;

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

    // interp our z buffering
    let v0z_loc = Point{x: v0.point.y, y: 1./v0.z as f64};
    let v1z_loc = Point{x: v1.point.y, y: 1./v1.z as f64};
    let v2z_loc = Point{x: v2.point.y, y: 1./v2.z as f64};

    // interp our vertex normals - must interpolate all 3 dimensions
    //V0
    let v0nx_loc = Point{x: v0.point.y, y: v0.normal.x as f64};
    let v0ny_loc = Point{x: v0.point.y, y: v0.normal.y as f64};
    let v0nz_loc = Point{x: v0.point.y, y: v0.normal.z as f64};

    //V1
    let v1nx_loc = Point{x: v1.point.y, y: v1.normal.x as f64};
    let v1ny_loc = Point{x: v1.point.y, y: v1.normal.y as f64};
    let v1nz_loc = Point{x: v1.point.y, y: v1.normal.z as f64};

    //V2
    let v2nx_loc = Point{x: v2.point.y, y: v2.normal.x as f64};
    let v2ny_loc = Point{x: v2.point.y, y: v2.normal.y as f64};
    let v2nz_loc = Point{x: v2.point.y, y: v2.normal.z as f64};

    let mut x01 = lerp_i32(&v0.point.reverse(), &v1.point.reverse());
    let mut z01 = lerp(&v0z_loc, &v1z_loc);

    //normals for segment V0 -> V1
    let mut nx01 = lerp(&v0nx_loc, &v1nx_loc);
    let mut ny01 = lerp(&v0ny_loc, &v1ny_loc);
    let mut nz01 = lerp(&v0nz_loc, &v1nz_loc);

    let x12 = lerp_i32(&v1.point.reverse(), &v2.point.reverse());
    let z12 = lerp(&v1z_loc, &v2z_loc);

    //normals for segment V1 -> V2
    let nx12 = lerp(&v1nx_loc, &v2nx_loc);
    let ny12 = lerp(&v1ny_loc, &v2ny_loc);
    let nz12 = lerp(&v1nz_loc, &v2nz_loc);

    let x02 = lerp_i32(&v0.point.reverse(), &v2.point.reverse()); //tall
    let z02 = lerp(&v0z_loc, &v2z_loc);

    //normals for segment V0 -> V2
    let nx02 = lerp(&v0nx_loc, &v2nx_loc);
    let ny02 = lerp(&v0ny_loc, &v2ny_loc);
    let nz02 = lerp(&v0nz_loc, &v2nz_loc);

    x01 = x01[0..x01.len() - 1].to_vec(); //remove last for concatenating
    z01 = z01[0..z01.len() - 1].to_vec();
    nx01 = nx01[0..nx01.len() - 1].to_vec();
    ny01 = ny01[0..ny01.len() - 1].to_vec();
    nz01 = nz01[0..nz01.len() - 1].to_vec();

    let x012 = [x01, x12].concat();
    let z012 = [z01, z12].concat();

    //normals concatenated side
    let nx012 = [nx01, nx12].concat();
    let ny012 = [ny01, ny12].concat();
    let nz012 = [nz01, nz12].concat();

    let xleft: Vec<i32>;
    let xright: Vec<i32>;

    let zleft: Vec<f64>;
    let zright: Vec<f64>;

    let nxleft: Vec<f64>;
    let nyleft: Vec<f64>;
    let nzleft: Vec<f64>;

    let nxright: Vec<f64>;
    let nyright: Vec<f64>;
    let nzright: Vec<f64>;

    let point = x02.len() / 2;
    if x02[point] < x012[point]  {
        xleft = x02;
        xright = x012;
        zleft = z02;
        zright = z012;

        nxleft = nx02;
        nyleft = ny02;
        nzleft = nz02;
        nxright = nx012;
        nyright = ny012;
        nzright = nz012;
    } else {
        xleft = x012;
        xright = x02;
        zleft = z012;
        zright = z02;

        nxleft = nx012;
        nyleft = ny012;
        nzleft = nz012;
        nxright = nx02;
        nyright = ny02;
        nzright = nz02;
    }

    for y in v0.point.y..v2.point.y+1 {
        let cur_y = (y - v0.point.y) as usize;

        let xl = xleft[cur_y];
        let xr = xright[cur_y];

        let z_segment = lerp(&Point{x: xl, y: zleft[cur_y]}, &Point{x: xr, y: zright[cur_y]});    
        
        let nx_segment = lerp(&Point{x: xl, y: nxleft[cur_y]}, &Point{x: xr, y: nxright[cur_y]});
        let ny_segment = lerp(&Point{x: xl, y: nyleft[cur_y]}, &Point{x: xr, y: nyright[cur_y]});
        let nz_segment = lerp(&Point{x: xl, y: nzleft[cur_y]}, &Point{x: xr, y: nzright[cur_y]});
        for x in xl..xr+1 {
            let segment_index = (x - xl) as usize;
            let z = z_segment[segment_index] as f32; //z == 1/z

            let (_x, _y) = canvas_to_zero_ind(x, y);
            //println!("{x},{y} -> {_x},{_y}");
            let z_ind = (_y * CANVAS_WIDTH as i32 + _x) as usize;
            if z > z_buffer[z_ind] {
                //CALCULATE
                let w_x = x as f32 / (VIEWPORT_Z_DIST * z);
                let w_y = y as f32 / (VIEWPORT_Z_DIST * z);
                let w_z = 1. / z;

                let normal = Vec3::new(nx_segment[segment_index] as f32, ny_segment[segment_index] as f32, nz_segment[segment_index] as f32);
                let world_pos = Vec3::new(w_x, w_y, w_z); //interp world pos

                let illumination = compute_illumination(&scene.lights, &scene.camera.position, &world_pos, &normal, &triangle.specular);
                let color_channels = v_rowmult(&color_to_float_channels(&triangle.color), &illumination);
                let final_color = float_channels_to_color(&color_channels);
                
                put_pixel(canvas, x, y, final_color);
                z_buffer[z_ind] = z;
            }
        }
    }
}


// Point3: x,y,h, s.t. h corresponds to color intensity at vertex given vertex vi
pub fn draw_triangle_shaded(canvas: &mut Canvas<Window>, mut v0: PointC, mut v1: PointC, mut v2: PointC, color: Color) {

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
}

pub fn draw_wireframe_triangle (canvas: &mut Canvas<Window>, z_buffer: &mut Vec<f32>, v0: DepthPoint, v1: DepthPoint, v2: DepthPoint, color: Color) {
    draw_line(canvas, z_buffer, v0, v1, color); //leftmost top-bottom line
    draw_line(canvas, z_buffer, v1, v2, color); //rightmost top-bottom line
    draw_line(canvas, z_buffer, v2, v0, color);
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
pub fn lerp<T, C>(point_i: &Point<T, C>, point_f: &Point<T, C>) -> Vec<C>
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
pub fn lerp_i32(point_i: &Point<i32, i32>, point_f: &Point<i32, i32>) -> Vec<i32> {
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
pub fn put_pixel(canvas: &mut Canvas<Window>, _x: i32, _y: i32, color: Color) {
    let (x, y) = canvas_to_zero_ind(_x, _y);
    if !in_range(x, 0, crate::CANVAS_WIDTH as i32 - 1) || !in_range(y, 0, crate::CANVAS_HEIGHT as i32 - 1) {
        println!("attempt put pixel {x}, {y} failed");
        return;
    }
    
    canvas.set_draw_color(color);
    match canvas.draw_point(sdl2::rect::Point::from((x, y))) {
        Ok(_) => {},
        Err(_) => {println!("Canvas error in put pixel for ({x},{y})");},
    };
}
/// convert from 0-centered 'indexing' to memory-layout (0-index layout)
#[inline]
pub fn canvas_to_zero_ind(x: i32, y: i32) -> (i32, i32) {
    ((((crate::CANVAS_WIDTH as f32/2.) + x as f32).round() as i32), (((crate::CANVAS_HEIGHT as f32/2.) - y as f32).round() as i32))
}

// color helpers
#[inline]
pub fn color_to_float_channels(color: &Color) -> Vec3 {
    Vec3::new(color.r as f32, color.g as f32, color.b as f32)
}

#[inline]
pub fn float_channels_to_8int(channels: &Vec3) -> (u8, u8, u8) {
    (channels[0].round() as u8, channels[1].round() as u8, channels[2].round() as u8)
}

#[inline]
pub fn float_channels_to_color(channels: &Vec3) -> Color {
    Color::from(float_channels_to_8int(channels))
}

#[inline]
pub fn clamp_point(p: &mut DepthPoint, x_range: (i32, i32), y_range: (i32, i32)) {
    assert!(x_range.0 < x_range.1 && y_range.0 < y_range.1);    
    let pp = &mut p.point;
    if pp.x < x_range.0 {
        pp.x = x_range.0;
    } else if pp.x > x_range.1 {
        pp.x = x_range.1;
    }

    if pp.y < y_range.0 {
        pp.y = y_range.0;
    } else if pp.y > y_range.1 {
        pp.y = y_range.1;
    }
}

// f32 helpers
// returns true if v is within [min, max]
#[inline]
pub fn in_range<T: Num + std::cmp::PartialOrd>(v: T, min: T, max: T) -> bool { v >= min && v <= max } 

// VEC3 math helpers
#[inline]
pub fn v_rowmult(u: &Vec3, v: &Vec3) -> Vec3 { Vec3::new(u[0] * v[0], u[1] * v[1], u[2] * v[2]) }
// invert row-wise, (1/v_0, 1/v_1, 1/v_2)
#[inline]
pub fn v_rowinvert(u: &Vec3) -> Vec3 { Vec3::new(1./u[0], 1./u[1], 1./u[2]) }
