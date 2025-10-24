#version 330

out vec4 outColor;

//texture coordinates: normalized fragment coordinates
varying vec2 texCoord;
			
// uniform variables of this render pass
//time 			
uniform float fTime;
uniform int frame;
//mouse position
uniform vec2 vMouseCoord;
//viewport size
uniform vec2 vResolution;
//camera position
uniform vec4 vViewPosition;

uniform sampler2D envMap; // Environment map texture

// GUI elements   // TODO add more if needed
uniform int averageCount;
uniform int maxBounces;
// How much a certain roughness scatters light
uniform float scattering_amplitude;
uniform bool spacial_jitter;
uniform vec3 skyLight;
uniform bool skyLightSwitch;
uniform bool showGround;
uniform bool showSphere;
uniform bool showWalls;
uniform bool showPillars;
uniform float wallRoughness;
uniform bool useEnvMap;
uniform bool colorByNormal;
uniform bool useGammaCorrection;
uniform bool showLocalLight;
uniform vec3 localLightColor;
uniform float animationSpeed;

// some basic transformations
mat4 Translate(in vec3 v) {       // TODO
		// Standard homogeneous translation matrix
        return mat4(
        vec4(1, 0, 0, 0),
        vec4(0, 1, 0, 0),
        vec4(0, 0, 1, 0),
        vec4(v, 1)
        );
}

mat4 RotateY(in float theta) {    // TODO
		// Standard homogeneous rotation matrix of positive theta degrees around the Y-axis
		float s = sin(theta);
		float c = cos(theta);
        return mat4(
        vec4(c, 0, -s, 0),
        vec4(0, 1, 0, 0),
        vec4(s, 0, c, 0),
        vec4(0, 0, 0, 1)
        );
}

mat4 Scale(in vec3 v) {          // TODO
		// Standard homogeneous scaling matrix
        return mat4(
        vec4(v.x, 0, 0, 0),
        vec4(0, v.y, 0, 0),
        vec4(0, 0, v.z, 0),
        vec4(0, 0, 0, 1)
        );
}

/* TODO: add other transforms if needed */

mat4 RotateX(in float theta) {    // TODO
		// Standard homogeneous rotation matrix of positive theta degrees around the X-axis
		float s = sin(theta);
		float c = cos(theta);
        return mat4(
        vec4(1, 0, 0, 0),
        vec4(0, c, -s, 0),
        vec4(0, s, c, 0),
        vec4(0, 0, 0, 1)
        );
}

mat4 RotateZ(in float theta) {    // TODO
		// Standard homogeneous rotation matrix of positive theta degrees around the Z-axis
		float s = sin(theta);
		float c = cos(theta);
        return mat4(
        vec4(c, -s, 0, 0),
        vec4(s, c, 0, 0),
        vec4(0, 0, 1, 0),
        vec4(0, 0, 0, 1)
        );
}



//--scene data---------------------------------------------------------------------

# define object types
const int TYPE_SPHERE=0;
const int TYPE_CUBE=1;

struct Object {
  bool enabled;
  int type;
  vec3 center;
  vec3 size;
  
  vec3 color;
  float roughness;
  bool isTransparent;
  bool isFloor;
  bool isLight;
  bool isSky;
  
  mat4 m2w;    // model to world matrix
  mat4 w2m;    // world to model matrix
  mat4 nv;     // normal vector transform 
};


#define NO 9         // TODO: make sure to match the number of objects NO with the objects you add in animateObjects().
Object objects[NO];

Object makeObject(bool enabled, int type, vec3 center, vec3 size, vec3 color, float roughness, mat4 T, mat4 R, mat4 S) {
    Object obj;
    obj.enabled=enabled;
    obj.type = type;
    obj.center = center;
    obj.size = size;
    obj.color = color;
    obj.roughness = roughness;
    obj.isTransparent = false;
    obj.isFloor = false;
    obj.isLight = false;
    obj.isSky = false;
    obj.m2w = T * R * S;  // TODO
    //obj.w2m = inverse(T * R * S);  // TODO
    
    // --- OPTIMISATION ---
    // Calculate inverses analytically
    
    // Inverse of Translate(v) is Translate(-v)
    mat4 T_inv = Translate(-center);
    
    // Inverse of Rotate(theta) is Rotate(-theta), which is its transpose
    mat4 R_inv = transpose(R); 
    
    // Inverse of Scale(v) is Scale(1.0 / v)
    vec3 invSize = 1.0 / size;
    mat4 S_inv = Scale(invSize);
    
    obj.w2m = S_inv * R_inv * T_inv;
    
    
    // For T * R * S, the 3x3 normal matrix simplifies to R * S_inv
    obj.nv = R * S_inv; 
    obj.nv[3] = vec4(0,0,0,1);
    
    return obj;
}


void animateObjects(float t) {    // ToDo: add three walls ...
    float pi = 3.1415926535897932;
	float animated_time = t * animationSpeed;
    
    {   // Sky
        vec3 center = vec3(0.0, 0.0, 0.0);
        vec3 size = vec3(10000.0); // surrounds whole scene
        vec3 color = skyLightSwitch?skyLight:vec3(0); 
        mat4 T = Translate(center);
        mat4 R = mat4(1.0);
        mat4 S = Scale(size);
        objects[0]=makeObject(true, TYPE_SPHERE, center, size, color,1, T, R, S);
        objects[0].isLight=true;
        objects[0].isSky = true;
    }
    
    {   // Shiny sphere
        vec3 center = vec3(-3, 0.0, 5.0);
        vec3 size = vec3(1.5);
        vec3 color = vec3(0.8);
        float roughness = 0.0;
        mat4 T = Translate(center);
        mat4 R = mat4(1.0);
        mat4 S = Scale(size);
        objects[1]=makeObject(showSphere, TYPE_SPHERE, center, size, color, roughness, T, R, S);
    }

    {   //  not so shiny Ground plane
        vec3 center = vec3(0.0, -2.4, 0.0);
        vec3 size = vec3(7.0,0.4,18.0);
        vec3 color = vec3(1,1,0);
        float roughness = 1.0;
        mat4 T = Translate(center);
        mat4 R = mat4(1.0);
        mat4 S = Scale(size);
        objects[2]=makeObject(showGround, TYPE_CUBE, center, size, color, roughness, T, R, S);
        objects[2].isFloor = true; 
    }
    
    {   // Local Light 1
        vec3 center = vec3(0.0, 8, 0.0);
        vec3 size = vec3(3.0,0.2,5.0);
        vec3 color = localLightColor;
        float roughness = 1.0; //does not matter for lights
        mat4 T = Translate(center);
        mat4 R = mat4(1.0);
        mat4 S = Scale(size);
        objects[3]=makeObject(showLocalLight, TYPE_CUBE, center, size, color, roughness, T, R, S);
        objects[3].isLight = true; 
    }
    
    {   // Back wall
        vec3 center = vec3(0.0, 3.0, 13.0);
        vec3 size = vec3(6, 6.0, 0.4);
        vec3 color = vec3(0.5,0.5,0.5);
        float roughness = wallRoughness;
        mat4 T = Translate(center);
        mat4 R = mat4(1.0);
        mat4 S = Scale(size);
        objects[4]=makeObject(showWalls, TYPE_CUBE, center, size, color, roughness, T, R, S);
        objects[4].isFloor = false; 
    }
    
    {   // Left wall
        vec3 center = vec3(-6.0, 3.0, 0.0);
        vec3 size = vec3(0.4, 6.0, 13.0);
        vec3 color = vec3(0,1,0);
        float roughness = wallRoughness;
        mat4 T = Translate(center);
        mat4 R = mat4(1.0);
        mat4 S = Scale(size);
        objects[5]=makeObject(showWalls, TYPE_CUBE, center, size, color, roughness, T, R, S);
        objects[5].isFloor = false; 
    }
    
    {   // Right wall
        vec3 center = vec3(6.0, 3.0, 0.0);
        vec3 size = vec3(0.4, 6.0, 13.0);
        vec3 color = vec3(1,0,0);
        float roughness = wallRoughness;
        mat4 T = Translate(center);
        mat4 R = mat4(1.0);
        mat4 S = Scale(size);
        objects[6]=makeObject(showWalls, TYPE_CUBE, center, size, color, roughness, T, R, S);
        objects[6].isFloor = false; 
    }
    
    {   // Pillar 1
        vec3 center = vec3(-1.0, 1.0, 10.0);
        vec3 size = vec3(1.0, 3.0, 1.0);
        vec3 color = vec3(0.5,0.5,0.5);
        float roughness = 0.0f;
        mat4 T = Translate(center);
        mat4 R = RotateY(radians(45.f));
        mat4 S = Scale(size);
        objects[7]=makeObject(showPillars, TYPE_CUBE, center, size, color, roughness, T, R, S);
        objects[7].isFloor = false; 
    }
    
    {   // Pillar 2
        vec3 center = vec3(3.0, 1.0, 10.0);
        vec3 size = vec3(1.0, 3.0, 1.0);
        vec3 color = vec3(0.5,0.5,0.5);
        float roughness = 0.0f;
        mat4 T = Translate(center);
        mat4 R = RotateY(animated_time);
        mat4 S = Scale(size);
        objects[8]=makeObject(showPillars, TYPE_CUBE, center, size, color, roughness, T, R, S);
        objects[8].isFloor = false; 
    }
    
    // TODO: add objects here when needed
}

// Creates a local coordinate system (U, V) around the normal (W)
void createOrthoNormalBasis(in vec3 W, out vec3 U, out vec3 V) {
    // Find a vector that's not parallel to W
    vec3 T = (abs(W.x) > 0.9) ? vec3(0, 1, 0) : vec3(1, 0, 0);
    
    // U is the "right" vector
    U = normalize(cross(T, W));
    
    // V is the "forward" vector
    V = cross(W, U);
}

// Converts a 3D direction vector to 2D UV coordinate for the sky map using "Equirectangular" mapping
vec2 getEnvMapUV(vec3 dir) {
    vec3 normDir = normalize(dir);
    float u = 0.5 + atan(normDir.z, normDir.x) / (2.0 * radians(180.f));
    float v = 0.5 + asin(normDir.y) / radians(180.f);
    return vec2(u, v);
}

/* returns signed distances (t0,t1) to the intersections of ray and sphere, t0 <= t1; 
   if there are no intersections (-1,-1) is returned.
*/
vec2 rayUnitSphereIntersect(vec3 r0, vec3 rd) { // TODO
	// Using the ray-sphere intersection algorithm provided during the lecture
	
	// The sphere has the origin as its center, so (r0 - C) = r0. Radius = 1
	float b = dot(rd, r0);
	float c = dot(r0, r0) - 1.f;
	float disc = b * b - c;
	
	// No intersection
	if ((disc) < 0.f) {
		return vec2(-1.f);
	}
	
	// Should work for only one intersection
	float t_plus = -b + sqrt(disc);
	float t_min = -b - sqrt(disc);
	
	// Since the square root will always be positive (or zero), t_min is always
	// either smaller or equal to t_plus
	return vec2(t_min, t_plus);
}

/* returns signed distances (t0,t1) to the intersections of ray and cube, t0 <= t1; 
   if there are no intersections (-1,-1) is returned.
*/
vec2 rayUnitBoxIntersect(vec3 r0, vec3 rd) {
	// Precalc inverse direction
	vec3 invDir = 1.0 / rd;

    // Compute intersection distances with the cube face planes
    vec3 tp = (vec3(-1) - r0) * invDir; // Postive faces
    vec3 tn = (vec3(1) - r0) * invDir; // Negative faces

    // Ensure tmin < tmax per axis
    // Unfortunately I don't think this works if multiple t-values are negative, i.e.
    // the cube is not entirely in front of the camera. In that case the most negative number
    // is chosen as tmin, instead of the least negative one, which is the one closest
    // to the camera. It shouldn't matter as negative t-values are discarded later anyway
    vec3 tmin = min(tp, tn);
    vec3 tmax = max(tp, tn);

    // The largest tmin is the entry point, while the smallest tmax is the exit point
    // This once again probably doesn't work for cubes not entirely in front of the camera, but
    // it shouldn't matter as t0 will be discarded in that case anyway
    float t0 = max(max(tmin.x, tmin.y), tmin.z);
    float t1  = min(min(tmax.x, tmax.y), tmax.z);

    // No intersection if t0 > t1
    if (t0 > t1)
        return vec2(-1);

    return vec2(t0, t1);

}

/* returns normal for unit cube surface point p */
vec4 computeCubeNormal(vec3 p) {
    vec3 absP = abs(p);
    if (absP.x > absP.y && absP.x > absP.z)
        return vec4(sign(p.x), 0.0, 0.0,0.0);
    else if (absP.y > absP.x && absP.y > absP.z)
        return vec4(0.0, sign(p.y), 0.0,0.0);
    else
        return vec4(0.0, 0.0, sign(p.z),0.0);
}

//-----------------------------------------------------------------------
struct HitData
{   
    float closest;     // distance to closest intersection 
    vec4 intersection; // point in model space
    Object obj;        // the object that has been hit
};

HitData AllObjectsRayTest(vec3 rayPos, vec3 rayDir)
{
    HitData hitData;
    hitData.closest = 999999.0; //keeps track of distance to closest intersection

	// Originally animate objects was called here which means it gets called every pixelm sample and bounce
	// We moved it to begining of main. 
	// does not cause a noticeable improvement in performance, but for large scenes it might
	//animateObjects(fTime);
	
    //all objects
    for(int i = 0; i < NO; i++)
    {
    	Object obj = objects[i];
    	
    	if (!obj.enabled) continue;
        
         // TODO 1: transform the ray to model space
	    vec4 mr0 = obj.w2m * vec4(rayPos,1); 
		vec4 mrd = obj.w2m * vec4(rayDir,0);
			
		// Store ray length to convert t back to world-space
        float mrd_length = length(mrd.xyz);
        
        // Normalise Ray Dir
        mrd.xyz = normalize(mrd.xyz);

        // compute the ray parameters t0t1 for entry and exit in model space
        vec2 t0t1 = obj.type==TYPE_SPHERE
        		? rayUnitSphereIntersect(mr0.xyz,mrd.xyz)
        		: rayUnitBoxIntersect(mr0.xyz,mrd.xyz); 
        
        // TODO 2: determine closest positive ray parameter for an intersection
        
        // Default to no intersection
        float t = -1.f;
        // The intersection functions ensure t0t1.x < t0t1.y, so if t0t1.x > 0, it's the closest
        // positive ray parameter
        if (t0t1.x > 0) {
			t = t0t1.x;
        }
        // t0t1.x is negative and t0t1.y positive, automatically making t0t1.y the closest
        // positive ray parameter
        else if (t0t1.y > 0) {
			t = t0t1.y;
        }
        // At this point, both ray parameters must be negative, 
        // and therefore no intersection takes place
        
        // Store model-space t
        float t_model = t;
        
        // Convert t to world-space
        t /= mrd_length;
        
        // TODO If new intersection is closer, update hit information
        if( t > 0.001 && t < hitData.closest)
        {
        	// The distance to the new closest intersection
            hitData.closest = t;
            // Update the object that is being hit the closest
			hitData.obj = obj;
			// The point in model space hit by the ray, using the vector equation of a line
		    hitData.intersection = mr0 + mrd * t_model; // note we are using model space t here
        }
    }    
    
    //all test finished, return shortest(best) hit data
    return hitData;
}

//--random functions: provided here because GLSL does not have one
float rand01(float seed) { return fract(sin(seed)*43758.5453123); }


// Generates a random direction in the hemisphere around the normal W, with a cosine weight
vec3 cosineSampleHemisphere(in float r1, in float r2, in vec3 U, in vec3 V, in vec3 W) {
    float pi = 3.14159265;
    
    // Create a random point on a 2D disk
    float r = sqrt(r1);
    float theta = 2.0 * pi * r2;
    float x = r * cos(theta);
    float y = r * sin(theta);
    
    // Project that 2D point up onto the unit hemisphere
    // We implement Malley's method
    float z = sqrt(1.0 - r1); // r1 is r*r
    
    // Go from local space (x,y,z) back to world space using our basis
    return normalize(x * U + y * V + z * W);
}

//---------------------------------------------------------------------
vec3 calculateFinalColor(vec3 cameraPos, vec3 cameraRayDir, float AAIndex, float spatial_hash)
{
    //init
    vec3 finalColor = vec3(0.0);
    vec3 absorbMul = vec3(1.0);
    vec3 rayOrigin = cameraPos;
    vec3 rayDir = cameraRayDir;

        
    //recursion not available in GLSL; replace by looping 
    //until hitting any light source || ray bounces too many times
    for(int i = 0; i < maxBounces; i++)
    {
        //+0.0001 to prevent ray already hit @ start pos
        HitData h = AllObjectsRayTest(rayOrigin + rayDir * 0.0001, rayDir);
        
       // rays end at light source
	   if (h.obj.isLight) {	
	   		if (h.obj.isSky && useEnvMap) {
                // It's the sky sphere AND we want to use the map
                vec2 uv = getEnvMapUV(rayDir);
                finalColor = texture(envMap, uv).rgb * absorbMul;
            } else {
                finalColor = h.obj.color * absorbMul;
            }
            break;
	   }
                   
        // The new ray origin is the point of intersection in world space
        rayOrigin = (h.obj.m2w * h.intersection).xyz;
      	                           
        vec3 normal = vec3(0,0,0);
        
        // Compute normal based on object type
        if (h.obj.type == TYPE_SPHERE) {
            // For a unit sphere, the model-space normal is the normalised intersection point
            vec4 model_normal = vec4(normalize(h.intersection.xyz), 0.0);
            // Transform normal to world space
            normal = normalize((h.obj.nv * model_normal).xyz);
        } else { // TYPE_CUBE
            vec4 model_normal = computeCubeNormal(h.intersection.xyz);
            // Transform normal to world space
            normal = normalize((h.obj.nv * model_normal).xyz);
        }
        
		// Seeds for random numbers
        // We need unique seeds based on pixel (ray_hash), sample (AAIndex), and bounce (i) to avoid correlation patterns
        // We use different magic numbers for each seed to ensure they are different

        float seed_bounce_type = float(i) * 1.234 + AAIndex * 434.578 + spatial_hash;
        float seed_r1 = float(i) * 5.678 + AAIndex * 312.123 + spatial_hash;
        float seed_r2 = float(i) * 8.901 + AAIndex * 983.123 + spatial_hash;
        float seed_x = float(i) * 2.345 + AAIndex * 897.709 + spatial_hash;
        float seed_y = float(i) * 3.456 + AAIndex * 734.978 + spatial_hash;
        float seed_z = float(i) * 4.567 + AAIndex * 123.456 + spatial_hash;
               
        // Here we will decide whether to do a specular or diffuse bounce
        float bounce_type_rand = rand01(seed_bounce_type);
        
        if (bounce_type_rand < h.obj.roughness)  // Diffuse bounce
        {
           // Create the local coordinate system
            vec3 U, V;
            createOrthoNormalBasis(normal, U, V);
            
            // Get two new random numbers for our hemisphere direction
            float r1 = rand01(seed_r1);
            float r2 = rand01(seed_r2);
            
            // The new ray direction is a random cosine-weighted vector
            rayDir = cosineSampleHemisphere(r1, r2, U, V, normal);
        }
        else	// Specular bounce
        {
       		float random_angle_x = scattering_amplitude * h.obj.roughness * (rand01(seed_x) - 0.5);
        	float random_angle_y = scattering_amplitude * h.obj.roughness * (rand01(seed_y) - 0.5);
        	float random_angle_z = scattering_amplitude * h.obj.roughness * (rand01(seed_z) - 0.5);
            
            // Rotate the normal by the random angles. This makes physical sense, as we're
        	// directly simulating the non-uniformity of rough suface normals.
        	// For an object with zero roughness, the normal should stay the same
            vec3 perturbed_normal = mat3(RotateX(random_angle_x)) * mat3(RotateY(random_angle_y)) * mat3(RotateZ(random_angle_z)) * normal;
            
            // We obtain the new direction by reflecting the ray w.r.t the (possibly perturbed) normal
            rayDir = reflect(rayDir, perturbed_normal);
        }
       
      
        // every bounce absorbs some light (more bounces = darker scene)
        //     - contains a hack for the checkerboard pattern of the floor
        float intensityFactor = (!h.obj.isFloor) ? 1.0: mod(int(200+rayOrigin.x) + int(200+rayOrigin.z),2);
        absorbMul *= (intensityFactor * h.obj.color);
        
        // Performance optimisation strategy: Russian Roullete
		// Start terminating rays after 3 bounces
		if (i > 3) {
    		// Get the max component of the color
    		float p = max(absorbMul.r, max(absorbMul.g, absorbMul.b));
    
    		// Set a minimum survival chance to avoid 0/0
    		float survival_prob = max(0.1, p); 

    		float seed_rr = float(i) * 123.456 + AAIndex * 789.123 + spatial_hash;
    		if (rand01(seed_rr) > survival_prob) {
       			break; // Terminate this path
    		}
    
    		// Boost the energy of paths that survive
    		absorbMul /= survival_prob;
		}
        	   
        if (colorByNormal)  {   // debugging option to see your normals
       		finalColor = 0.5 * (normal + vec3(1.0)); // maps [-1,1] to [0,1]
       		break;
       	}
    }    
    return finalColor;
}



//-----------------------------------------------------------------------
vec4 computeColor( vec2 fragCoord )
{    
    vec2 uv = fragCoord	 * 2.0 - 1.0;//transform fragment coords from [0,1] to [-1,1]
    uv.x *= vResolution.x / vResolution.y; //aspect fix

    //allow camera movement along z axis
    vec3 cameraPos = vec3(0.0, 2.0, -40.0 * vViewPosition.z);    
    vec3 cameraCenter = vec3(0.0,2,0);
    
    //camera lookat direction
    vec3 cameraDir = normalize(cameraCenter - cameraPos);
    
    //camera-to-fragment ray direction
    vec3 rayDir = normalize(3*cameraDir + vec3(uv, 0));
    
    // Trying to reduce correlation with random numbers
    float seed = dot(texCoord, vec2(12.9898, 78.233));
    float spatial_hash = rand01(seed);

    vec3 finalColor = vec3(0);
    for(int i = 1; i <= averageCount; i++)
    {
		if(spacial_jitter) {
			float seed1 = float(i) * 123.456 + spatial_hash;
        	float seed2 = float(i) * 789.123 + spatial_hash;
        
        	// Random offset from -0.5 to +0.5 pixels
        	float offsetX = (rand01(seed1) - 0.5) / vResolution.x;
        	float offsetY = (rand01(seed2) - 0.5) / vResolution.y;
        	
        	// Apply jitter to the normalized fragment coordinate
        	vec2 jitteredFragCoord = texCoord + vec2(offsetX, offsetY);
        	
        	// Recalculate UV and ray direction FOR EACH sample
        	vec2 uv = jitteredFragCoord * 2.0 - 1.0;
        	uv.x *= vResolution.x / vResolution.y;
        	rayDir = normalize(3*cameraDir + vec3(uv, 0));
		}
    
    
        finalColor += calculateFinalColor(cameraPos, rayDir, float(i), spatial_hash);
    }
    finalColor = finalColor/float(averageCount);//brute force AA & denoise
    
    if (useGammaCorrection)
    	finalColor.rgb = pow(finalColor.rgb,vec3(1.0/2.2));//gamma correction
        
    //result
    return vec4(finalColor.rgb, 1.0);
}


void main(void)
{
	animateObjects(fTime);

    gl_FragColor = computeColor(texCoord);
}

