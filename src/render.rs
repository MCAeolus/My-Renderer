use std::ops::AddAssign;
use num_traits::{ToPrimitive, Num};
use sdl2::{render::Canvas, video::Window, pixels::Color};
use crate::{structs::{Point, PointC, Reverse, Vec3}, CANVAS_WIDTH, CANVAS_HEIGHT};

pub fn draw_line(canvas: &mut Canvas<Window>, mut p1: Point<i32, i32>, mut p2: Point<i32, i32>, color: Color) {
    clamp_point(&mut p1, (-(CANVAS_WIDTH as i32)/2, CANVAS_WIDTH as i32/2 - 1), (-(CANVAS_HEIGHT as i32)/2, CANVAS_HEIGHT as i32/2 - 1));
    clamp_point(&mut p2, (-(CANVAS_WIDTH as i32/2), CANVAS_WIDTH as i32/2 - 1), (-(CANVAS_HEIGHT as i32)/2, CANVAS_HEIGHT as i32/2 - 1));
    if (p2.x - p1.x).abs() > (p2.y - p1.y).abs() { //horizontal-ish
        if p1.x > p2.x {
            swap(&mut p1, &mut p2);
        }
        let interps_y = lerp_i32(&p1, &p2);
        //dbg!(p1.x, p2.x);
        for x in p1.x..p2.x+1 {
            put_pixel(canvas, x, interps_y[(x - p1.x) as usize], color)
        }

    } else { //vertical-ish
        if p1.y > p2.y {
            swap(&mut p1, &mut p2);
        }
        let interps_x = lerp_i32(&p1.reverse(), &p2.reverse());
        //dbg!(p1.y, p2.y);
        for y in p1.y..p2.y+1 {
            put_pixel(canvas, interps_x[(y - p1.y) as usize], y, color)
        }
    }

}

pub fn draw_triangle(canvas: &mut Canvas<Window>, mut v0: Point<i32, i32>, mut v1: Point<i32, i32>, mut v2: Point<i32, i32>, color: Color) {

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

    //draw_line(canvas, v0, v1, Color::GREEN); //leftmost top-bottom line
    //draw_line(canvas, v1, v2, Color::BLUE); //rightmost top-bottom line
    //draw_line(canvas, v2, v0, Color::RED);
    


    //for y in min_y..max_y+1 {
    //    draw_line(canvas, Point::from());
    //}
}

pub fn draw_wireframe_triangle (canvas: &mut Canvas<Window>, v0: Point<i32, i32>, v1: Point<i32, i32>, v2: Point<i32, i32>, color: Color) {
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
    if !in_range(x, 0, crate::CANVAS_WIDTH as i32 -1) || !in_range(y, 0, crate::CANVAS_HEIGHT as i32-1) {
        //dbg!("attempt put pixel", x, y, "failed");
        return;
    }
    
    canvas.set_draw_color(color);
    match canvas.draw_point(sdl2::rect::Point::from((x, y))) {
        Ok(_) => {},
        Err(_) => {dbg!("Error in put pixel for ({x},{y})");},
    };
}
/// convert from 0-centered 'indexing' to memory-layout (0-index layout)
#[inline]
pub fn canvas_to_zero_ind(x: i32, y: i32) -> (i32, i32) {
    (((crate::CANVAS_WIDTH as f32/2. + x as f32).round() as i32), ((crate::CANVAS_HEIGHT as f32/2. - y as f32).round() as i32))
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
pub fn clamp_point(p: &mut Point<i32, i32>, x_range: (i32, i32), y_range: (i32, i32)) {
    assert!(x_range.0 < x_range.1 && y_range.0 < y_range.1);    
    if p.x < x_range.0 {
        p.x = x_range.0;
    } else if p.x > x_range.1 {
        p.x = x_range.1;
    }

    if p.y < y_range.0 {
        p.y = y_range.0;
    } else if p.y > y_range.1 {
        p.y = y_range.1;
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
// get vector length -> this could be amortized by calculating upfront (?) or lazy calculating

