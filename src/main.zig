const std = @import("std");

// Utility functions
const infinity = std.math.inf(f64);
const pi: f64 = 3.1415926535897932385;

fn degreesToRadians(degrees: f64) f64 {
    return degrees * pi / 180.0;
}

fn randomDouble(rand_gen: *std.rand.DefaultPrng) f64 {
    return rand_gen.random().float(f64);
}

fn randomDoubleInRange(rand_gen: *std.rand.DefaultPrng, min: f64, max: f64) f64 {
    return min + (max - min) * randomDouble(rand_gen);
}

fn clamp(x: f64, min: f64, max: f64) f64 {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

const Vec3 = struct {
    x: f64,
    y: f64,
    z: f64,

    fn initRandom(gen: *std.rand.DefaultPrng) Vec3 {
        return Vec3{
            .x = randomDouble(gen),
            .y = randomDouble(gen),
            .z = randomDouble(gen),
        };
    }

    fn initRandomInRange(gen: *std.rand.DefaultPrng, min: f64, max: f64) Vec3 {
        return Vec3{
            .x = randomDoubleInRange(gen, min, max),
            .y = randomDoubleInRange(gen, min, max),
            .z = randomDoubleInRange(gen, min, max),
        };
    }

    fn plus(self: Vec3, other: Vec3) Vec3 {
        return Vec3{
            .x = self.x + other.x,
            .y = self.y + other.y,
            .z = self.z + other.z,
        };
    }

    fn times(self: Vec3, scalar: f64) Vec3 {
        return Vec3{
            .x = self.x * scalar,
            .y = self.y * scalar,
            .z = self.z * scalar,
        };
    }

    fn pointwiseMultiply(self: Vec3, other: Vec3) Vec3 {
        return Vec3{
            .x = self.x * other.x,
            .y = self.y * other.y,
            .z = self.z * other.z,
        };
    }

    fn cross(self: Vec3, other: Vec3) Vec3 {
        return Vec3{
            .x = self.y * other.z - self.z * other.y,
            .y = self.z * other.x - self.x * other.z,
            .z = self.x * other.y - self.y * other.x,
        };
    }

    fn dot(self: Vec3, other: Vec3) f64 {
        return self.x * other.x + self.y * other.y + self.z * other.z;
    }

    fn lengthSquared(self: Vec3) f64 {
        return self.dot(self);
    }

    fn length(self: Vec3) f64 {
        return std.math.sqrt(lengthSquared(self));
    }

    fn minus(self: Vec3, other: Vec3) Vec3 {
        return self.plus(other.times(-1));
    }

    fn div_by(self: Vec3, scalar: f64) Vec3 {
        return self.times(1.0 / scalar);
    }

    fn unit(self: Vec3) Vec3 {
        return self.div_by(self.length());
    }

    fn nearZero(self: Vec3) bool {
        const s = 1e-8;
        return (@fabs(self.x) < s) and (@fabs(self.y) < s) and (@fabs(self.z) < s);
    }
};

fn makeRandomVecInUnitSphere(gen: *std.rand.DefaultPrng) Vec3 {
    while (true) {
        const p = Vec3.initRandomInRange(gen, -1, 1);
        if (p.lengthSquared() < 1) {
            return p;
        }
    }
}

fn randomUnitVector(gen: *std.rand.DefaultPrng) Vec3 {
    return makeRandomVecInUnitSphere(gen).unit();
}

const Point3 = Vec3;
const Color = Vec3;

const Ray = struct {
    origin: Point3,
    direction: Vec3,

    fn at(ray: Ray, t: f64) Point3 {
        return ray.origin.plus(ray.direction.times(t));
    }
};

fn writeColor(out: anytype, writer: anytype, pixel_color: Color, samples_per_pixel: i32) !void {
    // Divide the color by the number of samples and gamma-correct for gamma =
    // 2.0.
    const scale = 1.0 / @intToFloat(f32, samples_per_pixel);
    const r = std.math.sqrt(pixel_color.x * scale);
    const g = std.math.sqrt(pixel_color.y * scale);
    const b = std.math.sqrt(pixel_color.z * scale);

    const ir = @floatToInt(i32, 256 * clamp(r, 0.0, 0.999));
    const ig = @floatToInt(i32, 256 * clamp(g, 0.0, 0.999));
    const ib = @floatToInt(i32, 256 * clamp(b, 0.0, 0.999));

    try out.print("{} {} {}\n", .{ ir, ig, ib });
    try writer.flush();
}

fn rayColor(ray: Ray, world: World, gen: *std.rand.DefaultPrng, depth: i32) Color {
    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (depth <= 0) {
        return Color{ .x = 0, .y = 0, .z = 0 };
    }

    if (hitWorld(world, ray, 0.001, infinity)) |rec| {
        const maybe_scattered_ray = switch (rec.material) {
            .lambertian => |m| m.scatter(gen, ray, rec),
            .metal => |m| m.scatter(gen, ray, rec),
            .dielectric => |m| m.scatter(gen, ray, rec),
        };
        if (maybe_scattered_ray) |scattered_ray| {
            // Recursively gets the color of the place that the ray hits after
            // bouncing off the initial hit point rec.p.
            return rayColor(
                scattered_ray.ray,
                world,
                gen,
                depth - 1,
            ).pointwiseMultiply(scattered_ray.attenuation);
        } else {
            return Color{ .x = 0, .y = 0, .z = 0 };
        }
    } else {
        const unit_direction = ray.direction.unit();
        const t = 0.5 * (unit_direction.y + 1.0);
        return (Color{ .x = 1.0, .y = 1.0, .z = 1.0 }).times(1.0 - t)
            .plus((Color{ .x = 0.5, .y = 0.7, .z = 1.0 }).times(t));
    }
}

const HitRecord = struct {
    p: Point3,
    normal: Vec3,
    t: f64,
    front_face: bool,
    // TODO: Put this somewhere better so it doesn't need to be undefined.
    // Possible option would be to group the above fields in a HitGeometry
    // struct so that it can be constructed separately, but I don't feel like
    // doing that right now.
    material: Material = undefined,
};

const Sphere = struct { center: Point3, radius: f64 };

fn WorldObject(comptime Geometry: type) type {
    return struct { geometry: Geometry, material: Material };
}

const World = struct {
    spheres: std.ArrayList(WorldObject(Sphere)),
};

const ScatteredRay = struct {
    attenuation: Color,
    ray: Ray,
};

// Reflects v off a surface with normal vector n.
fn reflect(v: Vec3, n: Vec3) Vec3 {
    return v.minus(n.times(2 * v.dot(n)));
}

fn refract(uv: Vec3, n: Vec3, etai_over_etat: f64) Vec3 {
    const cos_theta = @min(uv.times(-1).dot(n), 1.0);
    const r_out_perp = uv.plus(n.times(cos_theta)).times(etai_over_etat);
    const r_out_parallel = n.times(-std.math.sqrt(@fabs(1.0 - r_out_perp.lengthSquared())));
    return r_out_perp.plus(r_out_parallel);
}

fn reflectance(cosine: f64, ref_idx: f64) f64 {
    // Use Schlick's approximation for reflectance.
    var r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * std.math.pow(f64, 1 - cosine, 5);
}

const Material = union(enum) {
    lambertian: struct {
        albedo: Color,

        fn scatter(self: @This(), gen: *std.rand.DefaultPrng, _: Ray, rec: HitRecord) ?ScatteredRay {
            const candidate_scatter_direction = rec.normal.plus(randomUnitVector(gen));
            const scatter_direction = if (candidate_scatter_direction.nearZero()) rec.normal else candidate_scatter_direction;
            return ScatteredRay{
                .attenuation = self.albedo,
                .ray = Ray{ .origin = rec.p, .direction = scatter_direction },
            };
        }
    },
    metal: struct {
        albedo: Color,
        fuzz: f64,

        fn scatter(self: @This(), gen: *std.rand.DefaultPrng, r_in: Ray, rec: HitRecord) ?ScatteredRay {
            const reflected = reflect(r_in.direction.unit(), rec.normal);
            const result = ScatteredRay{
                .attenuation = self.albedo,
                .ray = Ray{
                    .origin = rec.p,
                    .direction = reflected.plus(makeRandomVecInUnitSphere(gen).times(self.fuzz)),
                },
            };
            return if (result.ray.direction.dot(rec.normal) > 0) result else null;
        }
    },
    dielectric: struct {
        ir: f64, // Index of refraction

        fn scatter(self: @This(), gen: *std.rand.DefaultPrng, r_in: Ray, rec: HitRecord) ?ScatteredRay {
            const refraction_ratio = if (rec.front_face) (1.0 / self.ir) else self.ir;
            const unit_direction = r_in.direction.unit();
            const cos_theta = @min(unit_direction.times(-1).dot(rec.normal), 1.0);
            const sin_theta = std.math.sqrt(1.0 - cos_theta * cos_theta);
            const cannot_refract = refraction_ratio * sin_theta > 1.0;
            const direction = if (cannot_refract or reflectance(cos_theta, refraction_ratio) > randomDouble(gen)) reflect(unit_direction, rec.normal) else refract(unit_direction, rec.normal, refraction_ratio);

            return ScatteredRay{
                .attenuation = Color{ .x = 1.0, .y = 1.0, .z = 1.0 },
                .ray = Ray{ .origin = rec.p, .direction = direction },
            };
        }
    },
};

fn hitWorld(world: World, ray: Ray, t_min: f64, t_max: f64) ?HitRecord {
    var rec: ?HitRecord = null;
    var closest_so_far = t_max;

    for (world.spheres.items) |obj| {
        if (hitSphere(obj.geometry, ray, t_min, closest_so_far)) |temp_rec| {
            closest_so_far = temp_rec.t;
            rec = temp_rec;
            // TODO: Do something to avoid this seeming optional when it isn't.
            rec.?.material = obj.material;
        }
    }
    return rec;
}

fn hitSphere(sphere: Sphere, ray: Ray, t_min: f64, t_max: f64) ?HitRecord {
    const oc = ray.origin.minus(sphere.center);
    const a = ray.direction.lengthSquared();
    const half_b = oc.dot(ray.direction);
    const c = oc.lengthSquared() - sphere.radius * sphere.radius;
    const discriminant = half_b * half_b - a * c;
    if (discriminant < 0) {
        return null;
    } else {
        const sqrtd = std.math.sqrt(discriminant);
        var root = (-half_b - sqrtd) / a;
        if (root < t_min or t_max < root) {
            root = (-half_b + sqrtd) / a;
            if (root < t_min or t_max < root) {
                return null;
            }
        }
        const t = root;
        const p = ray.at(t);
        const outward_normal = p.minus(sphere.center).div_by(sphere.radius);
        const front_face = ray.direction.dot(outward_normal) < 0.0;
        const normal = if (front_face) outward_normal else outward_normal.times(-1);
        return HitRecord{ .p = p, .t = t, .normal = normal, .front_face = front_face };
    }
}

fn makeRandomVecInUnitDisk(gen: *std.rand.DefaultPrng) Vec3 {
    while (true) {
        const p = Vec3{
            .x = randomDoubleInRange(gen, -1, 1),
            .y = randomDoubleInRange(gen, -1, 1),
            .z = 0,
        };
        if (p.lengthSquared() >= 1) continue;
        return p;
    }
}

const Camera = struct {
    origin: Point3,
    lower_left_corner: Point3,
    horizontal: Vec3,
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    w: Vec3,
    lens_radius: f64,

    fn init(
        lookfrom: Point3,
        lookat: Point3,
        vup: Vec3,
        vfov: f64,
        aspect_ratio: f64,
        aperture: f64,
        focus_dist: f64,
    ) Camera {
        const theta = degreesToRadians(vfov);
        const h = std.math.tan(theta / 2);
        const viewport_height = 2.0 * h;
        const viewport_width = aspect_ratio * viewport_height;
        const w = lookfrom.minus(lookat).unit();
        const u = vup.cross(w).unit();
        const v = w.cross(u);
        const origin = lookfrom;
        const horizontal = u.times(focus_dist * viewport_width);
        const vertical = v.times(focus_dist * viewport_height);
        const lower_left_corner = origin
            .minus(horizontal.div_by(2))
            .minus(vertical.div_by(2))
            .minus(w.times(focus_dist));
        const lens_radius = aperture / 2;
        return Camera{
            .origin = origin,
            .lower_left_corner = lower_left_corner,
            .horizontal = horizontal,
            .vertical = vertical,
            .u = u,
            .v = v,
            .w = w,
            .lens_radius = lens_radius,
        };
    }

    fn getRay(self: @This(), s: f64, t: f64, gen: *std.rand.DefaultPrng) Ray {
        const rd = makeRandomVecInUnitDisk(gen).times(self.lens_radius);
        const offset = self.u.times(rd.x).plus(self.v.times(rd.y));

        return Ray{
            .origin = self.origin.plus(offset),
            .direction = self.lower_left_corner
                .plus(self.horizontal.times(s))
                .plus(self.vertical.times(t))
                .minus(self.origin)
                .minus(offset),
        };
    }
};

fn makeRandomScene(alloc: std.mem.Allocator, gen: *std.rand.DefaultPrng) !std.ArrayList(WorldObject(Sphere)) {
    var spheres = std.ArrayList(WorldObject(Sphere)).init(alloc);
    errdefer spheres.deinit();

    const ground_material = Material{
        .lambertian = .{ .albedo = Color{ .x = 0.5, .y = 0.5, .z = 0.5 } },
    };
    try spheres.append(.{
        .geometry = Sphere{ .center = Point3{ .x = 0, .y = -1000, .z = 0 }, .radius = 1000 },
        .material = ground_material,
    });

    {
        var a: i32 = -11;
        while (a < 11) : (a += 1) {
            var b: i32 = -11;
            while (b < 11) : (b += 1) {
                const choose_mat = randomDouble(gen);
                const center = Point3{
                    .x = @intToFloat(f64, a) + 0.9 * randomDouble(gen),
                    .y = 0.2,
                    .z = @intToFloat(f64, b) + 0.9 * randomDouble(gen),
                };

                if (center.minus(.{ .x = 4, .y = 0.2, .z = 0 }).length() > 0.9) {
                    const material = expr: {
                        if (choose_mat < 0.8) {
                            // Diffuse
                            break :expr Material{ .lambertian = .{ .albedo = Vec3.initRandom(gen) } };
                        } else if (choose_mat < 0.95) {
                            // Metal
                            break :expr Material{ .metal = .{
                                .albedo = Vec3.initRandomInRange(gen, 0.5, 1),
                                .fuzz = randomDoubleInRange(gen, 0, 0.5),
                            } };
                        } else {
                            // Glass
                            break :expr Material{
                                .dielectric = .{ .ir = 1.5 },
                            };
                        }
                    };

                    try spheres.append(.{
                        .geometry = Sphere{ .center = center, .radius = 0.2 },
                        .material = material,
                    });
                }
            }
        }
    }
    const material1 = Material{ .dielectric = .{ .ir = 1.5 } };
    try spheres.append(.{
        .geometry = Sphere{ .center = Point3{ .x = 0, .y = 1, .z = 0 }, .radius = 1.0 },
        .material = material1,
    });

    const material2 = Material{ .lambertian = .{ .albedo = Color{ .x = 0.4, .y = 0.2, .z = 0.1 } } };
    try spheres.append(.{
        .geometry = Sphere{ .center = Point3{ .x = -4, .y = 1, .z = 0 }, .radius = 1.0 },
        .material = material2,
    });

    const material3 = Material{ .metal = .{
        .albedo = Color{ .x = 0.7, .y = 0.6, .z = 0.5 },
        .fuzz = 0.0,
    } };

    try spheres.append(.{
        .geometry = Sphere{ .center = Point3{ .x = 4, .y = 1, .z = 0 }, .radius = 1.0 },
        .material = material3,
    });

    return spheres;
}

fn makeFixedScene(alloc: std.mem.Allocator) !std.ArrayList(WorldObject(Sphere)) {
    const material_ground = Material{
        .lambertian = .{ .albedo = Color{ .x = 0.8, .y = 0.8, .z = 0.0 } },
    };

    const material_center = Material{
        .lambertian = .{ .albedo = Color{ .x = 0.1, .y = 0.2, .z = 0.5 } },
    };
    const material_left = Material{
        .dielectric = .{ .ir = 1.5 },
    };
    const material_right = Material{
        .metal = .{ .albedo = Color{ .x = 0.8, .y = 0.6, .z = 0.2 }, .fuzz = 0.0 },
    };

    var spheres = std.ArrayList(WorldObject(Sphere)).init(alloc);
    errdefer spheres.deinit();
    try spheres.append(.{
        .geometry = Sphere{ .center = Point3{ .x = 0, .y = -100.5, .z = -1 }, .radius = 100 },
        .material = material_ground,
    });
    try spheres.append(.{
        .geometry = Sphere{ .center = Point3{ .x = 0, .y = 0, .z = -1 }, .radius = 0.5 },
        .material = material_center,
    });

    try spheres.append(.{
        .geometry = Sphere{ .center = Point3{ .x = -1, .y = 0, .z = -1 }, .radius = 0.5 },
        .material = material_left,
    });
    try spheres.append(.{
        .geometry = Sphere{ .center = Point3{ .x = -1, .y = 0, .z = -1 }, .radius = -0.4 },
        .material = material_left,
    });
    try spheres.append(.{
        .geometry = Sphere{ .center = Point3{ .x = 1, .y = 0, .z = -1 }, .radius = 0.5 },
        .material = material_right,
    });
    return spheres;
}

pub fn main() !void {
    // Image
    const aspect_ratio = 3.0 / 2.0;
    const image_width = 1200;
    const image_height = @floatToInt(i32, @intToFloat(f64, image_width) / aspect_ratio);
    const num_samples_per_pixel = 500;
    const lookfrom = Point3{ .x = 13, .y = 2, .z = 3 };
    const lookat = Point3{ .x = 0, .y = 0, .z = 0 };
    const vup = Vec3{ .x = 0, .y = 1, .z = 0 };
    const dist_to_focus = 10.0;
    const aperture = 0.1;

    const camera = Camera.init(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);

    const max_depth = 50;

    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const gpa = general_purpose_allocator.allocator();
    var gen = std.rand.DefaultPrng.init(@intCast(u64, std.time.timestamp()));

    //var spheres = try makeFixedScene(gpa);
    var spheres = try makeRandomScene(gpa, &gen);
    defer spheres.deinit();

    // World
    const world = World{
        .spheres = spheres,
    };

    // Render
    const stdout_file = std.io.getStdOut().writer();
    var bw = std.io.bufferedWriter(stdout_file);
    const stdout = bw.writer();

    try stdout.print("P3\n{} {}\n255\n", .{ image_width, image_height });
    try bw.flush(); // don't forget to flush!

    {
        var j: i32 = image_height - 1;
        while (j >= 0) : (j -= 1) {
            var i: i32 = 0;
            std.debug.print("\nScanlines remaining: {}\n", .{j});
            while (i < image_width) : (i += 1) {
                var pixel_color = Color{ .x = 0, .y = 0, .z = 0 };
                {
                    var s: u32 = 0;
                    while (s < num_samples_per_pixel) : (s += 1) {
                        const u =
                            (@intToFloat(f64, i) + randomDouble(&gen)) /
                            @intToFloat(f64, image_width - 1);
                        const v =
                            (@intToFloat(f64, j) + randomDouble(&gen)) /
                            @intToFloat(f64, image_height - 1);
                        // Ray from "eye" to screen.
                        const r = camera.getRay(u, v, &gen);
                        pixel_color = pixel_color.plus(rayColor(
                            r,
                            world,
                            &gen,
                            max_depth,
                        ));
                    }
                }
                try writeColor(stdout, &bw, pixel_color, num_samples_per_pixel);
            }
        }
    }
}
