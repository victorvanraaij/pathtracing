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

// GUI elements   // TODO add more if needed
 uniform int averageCount;
 uniform int maxBounces;
 uniform vec3 skyLight;
 uniform bool skyLightSwitch;
 uniform bool showGround;
 uniform bool showSphere;

 uniform bool colorByNormal;
 uniform bool useGammaCorrection;


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
  
  mat4 m2w;    // model to world matrix
  mat4 w2m;    // world to model matrix
  mat4 nv;     // normal vector transform 
};


#define NO 3         // TODO: make sure to match the number of objects NO with the objects you add in animateObjects().
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
    obj.m2w = mat4(1);  // TODO
    obj.w2m = mat4(1);  // TODO
    obj.nv  = mat4(1);  // TODO
    return obj;
}


void animateObjects(float t) {    // ToDo: add three walls ...
    float pi = 3.1415926535897932;

    
    {   // Sky
        vec3 center = vec3(0.0, 0.0, 0.0);
        vec3 size = vec3(10000.0); // surrounds whole scene
        vec3 color = skyLightSwitch?skyLight:vec3(0); 
        mat4 T = Translate(center);
        mat4 R = mat4(1.0);
        mat4 S = Scale(size);
        objects[0]=makeObject(true, TYPE_SPHERE, center, size, color,1, T, R, S);
        objects[0].isLight=true;
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
    
    // TODO: add objects here when needed
}

/* returns signed distances (t0,t1) to the intersections of ray and sphere, t0 <= t1; 
   if there are no intersections (-1,-1) is returned.
*/
vec2 rayUnitSphereIntersect(vec3 r0, vec3 rd) { // TODO
	// Using the ray-sphere intersection algorithm provided during the lecture
	
	// The sphere has the origin as its center, so (r0 - C) = r0. Radius = 1
	float b = dot(rd, r0);
	float c = dot(r0, r0) - 1;
	
	// No intersection
	if ((b * b - c) < 0) {
		return vec2(-1,-1);
	}
	
	// Should work for only one intersection
	float t_plus = -b + sqrt(b * b - c);
	float t_min = -b - sqrt(b * b - c);
	
	// Ensure the return vector is in the right order
	if (t_plus < t_min) {
		return vec2(t_plus, t_min);
	}
	else {
		return vec2(t_min, t_plus);
	}
}

/* returns signed distances (t0,t1) to the intersections of ray and cube, t0 <= t1; 
   if there are no intersections (-1,-1) is returned.
*/
vec2 rayUnitBoxIntersect(vec3 r0, vec3 rd) {   // TODO
	vec3 X = vec3(1, 0, 0);
	vec3 Y = vec3(0, 1, 0);
	vec3 Z = vec3(0, 0, 1);
	
	// First, we're computing the intersection of the ray with each of the six face 
	// planes of the cube, using the ray-plane algorithm provided during the lecture
	float t_pxy = (dot(Z, vec3(0, 0, 1) - r0)) / dot(Z, rd); // positive x,y-plane
	float t_nxy = (dot(Z, vec3(0, 0, -1) - r0)) / dot(Z, rd); // negative x,y-plane
	float t_pxz = (dot(Y, vec3(0, 1, 0) - r0)) / dot(Y, rd); // positive x,z-plane
	float t_nxz = (dot(Y, vec3(0, -1, 0) - r0)) / dot(Y, rd); // negative x,z-plane
	float t_pyz = (dot(X, vec3(1, 0, 0) - r0)) / dot(X, rd); // positive y,z-plane
	float t_nyz = (dot(X, vec3(-1, 0, 0) - r0)) / dot(X, rd); // negative y,z-plane
	
	return vec2(-1,-1);
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

	animateObjects(fTime);
	
    //all objects
    for(int i = 0; i < NO; i++)
    {
    	Object obj = objects[i];
    	
    	if (!obj.enabled) continue;
        
         // TODO 1: transform the ray to model space
	    vec4 mr0 = vec4(rayPos,1); 
		vec4 mrd = vec4(rayDir,0);
        

        // compute the ray parameters t0t1 for entry and exit in model space
        vec2 t0t1 = obj.type==TYPE_SPHERE
        		? rayUnitSphereIntersect(mr0.xyz,mrd.xyz)
        		: rayUnitBoxIntersect(mr0.xyz,mrd.xyz); 
        
        // TODO 2: determine closest positive ray parameter for an intersection
        float t = t0t1.x;
        
        // TODO If new intersection is closer, update hit information
        if( t > 0.001 && t < hitData.closest)
        {
            hitData.closest = 0;
			hitData.obj = obj;
		    hitData.intersection = mr0;
        }
    }    
    
    //all test finished, return shortest(best) hit data
    return hitData;
}

//--random functions: provided here because GLSL does not have one
float rand01(float seed) { return fract(sin(seed)*43758.5453123); }


//---------------------------------------------------------------------
vec3 calculateFinalColor(vec3 cameraPos, vec3 cameraRayDir, float AAIndex)
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
        
        return h.obj.color;   // TODO 1: REMOVE THIS LINE WHEN INSTRUCTED TO IN ASSIGNMENT TEXT
        
       // rays end at light source
	   if (h.obj.isLight) {	finalColor = h.obj.color * absorbMul;	break; }
                   
        //TODO 2:update rayOrigin for next bounce
        rayOrigin = rayOrigin;
      	                           
        //TODO 3: update rayDir for next bounce
        vec3 normal = vec3(0,0,0);
        rayDir = rayDir;	        
      
        // every bounce absorbs some light (more bounces = darker scene)
        //     - contains a hack for the checkerboard pattern of the floor
        float intensityFactor = (!h.obj.isFloor) ? 1.0: mod(int(200+rayOrigin.x) + int(200+rayOrigin.z),2);
        absorbMul *= (intensityFactor * h.obj.color);
        	   
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

    vec3 finalColor = vec3(0);
    for(int i = 1; i <= averageCount; i++)
    {
        finalColor += calculateFinalColor(cameraPos, rayDir, float(i));
    }
    finalColor = finalColor/float(averageCount);//brute force AA & denoise
    
    if (useGammaCorrection)
    	finalColor.rgb = pow(finalColor.rgb,vec3(1.0/2.2));//gamma correction
        
    //result
    return vec4(finalColor.rgb, 1.0);
}


void main(void)
{
    gl_FragColor = computeColor(texCoord);
}

