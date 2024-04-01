import glfw
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLUT.freeglut import *
from PIL import Image
import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from pygame import Vector3, Vector2
import random
from noise import pnoise2
import glm
import pyrr
import multiprocessing as mp
import pyfastnoisesimd as fns
import random


class Block():
    def __init__(self, id) -> None:
        self.id = id
        self.omitables = [False for _ in range(6)]
    
    def omit_faces(self, grid, x, y, z, adjacent_chunks):
        w = grid.w
        h = grid.h
        d = grid.d

        cid = self.id
        omitables = [False, False, False, False, False, False]  # [z+, z-, y+, y-, x+, x-]

        # Check within the current chunk
        if y + 1 < h and (grid.blocks[x][y + 1][z].id == cid or grid.blocks[x][y + 1][z].id != 0):
            omitables[2] = True
        if y > 0 and (grid.blocks[x][y - 1][z].id == cid or grid.blocks[x][y - 1][z].id != 0):
            omitables[3] = True
        if x + 1 < w and (grid.blocks[x + 1][y][z].id == cid or grid.blocks[x + 1][y][z].id != 0):
            omitables[4] = True
        if x > 0 and (grid.blocks[x - 1][y][z].id == cid or grid.blocks[x - 1][y][z].id != 0):
            omitables[5] = True
        if z + 1 < d and (grid.blocks[x][y][z + 1].id == cid or grid.blocks[x][y][z + 1].id != 0):
            omitables[0] = True
        if z > 0 and (grid.blocks[x][y][z - 1].id == cid or grid.blocks[x][y][z - 1].id != 0):
            omitables[1] = True

        # Check adjacent chunks
        north_chunk, south_chunk, east_chunk, west_chunk = adjacent_chunks

        # # North (positive y)
        # if z == d - 1:
        #     if not north_chunk:
        #         omitables[0] = True
        #     elif north_chunk.blocks[x][y][0].id == cid or north_chunk.blocks[x][y][0].id != 0:
        #         omitables[0] = True
        # # South (negative y)
        # if z == 0:
        #     if not south_chunk:
        #         omitables[1] = True
        #     elif south_chunk.blocks[x][y][d - 1].id == cid or south_chunk.blocks[x][y][d - 1].id != 0:
        #         omitables[1] = True
        # # East (positive x)
        # if x == w - 1:
        #     if not east_chunk:
        #         omitables[4] = True
        #     elif east_chunk.blocks[0][y][z].id == cid or east_chunk.blocks[0][y][z].id != 0:
        #         omitables[4] = True
        # # West (negative x)
        # if x == 0:
        #     if not west_chunk:
        #         omitables[5] = True
        #     elif west_chunk.blocks[w - 1][y][z].id == cid or west_chunk.blocks[w - 1][y][z].id != 0:
        #         omitables[5] = True
      
    

       

        self.omitables = omitables
class Chunk():
    def __init__(self, x, y, w, h, d, blocks, index) -> None:
        self.blocks = blocks
        self.w = w
        self.h = h
        self.d = d
        self.x = x
        self.y = y
        self.updated = False
        self.index = index
        self.vertex_data = []

class Camera:
    def __init__(self, position, look_at, up):
        self.position = position
        self.look_at = look_at
        self.up = up

    def move(self, direction, amount):
        if direction == "FORWARD":
            self.position += (self.look_at - self.position) * amount
        elif direction == "BACKWARD":
            self.position -= (self.look_at - self.position) * amount
        # Add more movement directions (LEFT, RIGHT, UP, DOWN) as needed

    def rotate(self, axis, angle):
        # This is a simplified way to rotate the camera around an axis
        # You might want to use quaternions or Euler angles for a more robust solution
        if axis == "YAW":
            # Implement yaw rotation
            pass
        elif axis == "PITCH":
            # Implement pitch rotation
            pass
        # Add more rotation axes as needed


# Vertex shader source code
vertex_shader_source = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec4 aColor;
layout (location = 2) in vec2 aTexCoord; // Texture coordinate

uniform mat4 view;
uniform mat4 projection;
out vec4 vertexColor; 
out vec2 TexCoord;

void main()
{
    gl_Position = projection * view * vec4(aPos, 1.0); // Simplified aPos usage
    vertexColor = aColor; // Pass color to the fragment shader correctly
    TexCoord = aTexCoord;
}
"""

# Fragment shader source code
fragment_shader_source = """
#version 330 core
out vec4 FragColor;
in vec4 vertexColor; // Received from vertex shader
in vec2 TexCoord;
uniform sampler2D texture1;
void main()
{
    FragColor = texture(texture1, TexCoord) * vertexColor; // Use the vertex color
}
"""
global sun_dir
sun_dir = Vector3(3, 3, -4).normalize()

def draw_quad(p1, p2, p3, p4, id):
    global sun_dir
    normal = (p2 - p1).cross(p3 - p1).normalize()
    light = normal.dot(sun_dir) / 3
    num_textures = 4
    tex_offset = 0
 
    tex_offset = (id - 1) / num_textures
    alpha = 1
  
    if id == 2:
        alpha = 0.5
    else:
        alpha = 1
   
    verts = [
        p1.x, p1.y, p1.z, 0.3 + light, 0.3 + light, 0.3 + light, alpha, 0 + tex_offset, 0, 
        p2.x, p2.y, p2.z, 0.3 + light, 0.3 + light, 0.3 + light, alpha, 1 / num_textures + tex_offset, 0,
        p4.x, p4.y, p4.z, 0.3 + light, 0.3 + light, 0.3 + light, alpha, 1 / num_textures + tex_offset, 1,
        p3.x, p3.y, p3.z, 0.3 + light, 0.3 + light, 0.3 + light, alpha, 0 + tex_offset, 1
    ]
    
    return verts

def draw_cube(pos, rot, omit, id):
    cube_faces = [
        [
            Vector3(-1, -1, 1),
            Vector3(-1, 1, 1),
            Vector3(1, -1, 1),
            Vector3(1, 1, 1)
        ],
            [
            Vector3(-1, -1, -1),
            Vector3(-1, 1, -1),
            Vector3(1, -1, -1),
            Vector3(1, 1, -1)
        ],
            [
            Vector3(-1, 1, -1),
            Vector3(-1, 1, 1),
            Vector3(1, 1, -1),
            Vector3(1, 1, 1)
        ],
            [
            Vector3(-1, -1, -1),
            Vector3(-1, -1, 1),
            Vector3(1, -1, -1),
            Vector3(1, -1, 1)
        ],
            [
            Vector3(1, -1, -1),
            Vector3(1, -1, 1),
            Vector3(1, 1, -1),
            Vector3(1, 1, 1)
        ],
            [
            Vector3(-1, -1, -1),
            Vector3(-1, -1, 1),
            Vector3(-1, 1, -1),
            Vector3(-1, 1, 1)
        ]
    ]
    list = []
    for i, f in enumerate(cube_faces):

        if omit[i] == True:
            continue
        p1, p2, p3, p4 = f[0], f[1], f[2], f[3]
  
        
        result = draw_quad(
            Vector3(p1.x, p1.y, p1.z).rotate(rot, Vector3(0, 1, 0)) + pos, 
            Vector3(p2.x, p2.y, p2.z).rotate(rot, Vector3(0, 1, 0)) + pos, 
            Vector3(p3.x, p3.y, p3.z).rotate(rot, Vector3(0, 1, 0)) + pos, 
            Vector3(p4.x, p4.y, p4.z).rotate(rot, Vector3(0, 1, 0)) + pos,
            id
        )
       
        list.extend(result)
    return list




def load_texture(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img_data = np.array(list(img.getdata()), np.uint8)
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    # glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    return texture_id


def generate_chunk(info):
    _x, _y, index = info

    w = 16
    d = 16
    h = 32


    grid = [[[Block(0) for _ in range(w)] for _ in range(h)] for _ in range(d)]

    for x in range(w):
        for y in range(h):
            for z in range(d):
                
                if (y < pnoise2((x + _x * w) / w, (z + _y * d) / d) * 40 + 10):
                    grid[x][y][z] = Block(3)
            
                elif y < 3:
                    if y == 0:
                        grid[x][y][z] = Block(3)
                    else:
                        grid[x][y][z] = Block(2)            
                
    for x in range(w):
        for y in range(h):
            for z in range(d):
                curr = grid[x][y][z]
                if curr.id == 3 and y < h - 1:
                    if grid[x][y + 1][z].id == 0:
                        if y > 3 or y < 2:
                            grid[x][y][z] = Block(1)
                        else:
                            grid[x][y][z] = Block(4)
                    elif grid[x][y + 1][z].id == 2:
                        grid[x][y][z] = Block(4)
                    

    new_chunk = Chunk(_x, _y, w, h, d, grid, index)
    new_chunk.updated = False

    return new_chunk
  



def main():
    # Initialize GLFW
    if not glfw.init():
        return
    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(2000, 1000, "Simple Triangle", None, None)
    if not window:
        glfw.terminate()
        print("Failed to create GLFW window")
        return
    global lastX, lastY, yaw, pitch, player_front, player_pos, player_front, camera_speed
    # Assuming these are at the beginning of your main() after initializing GLFW
    player_pos = Vector3(0.0, 0.0, 20.0)
    player_front = Vector3(0.0, 0.0, -1.0)
    camera_up = Vector3(0.0, 1.0, 0.0)

    yaw = -90.0  # Yaw is initialized to -90.0 degrees since a yaw of 0.0 results in a direction vector pointing to the right
    pitch = 0.0
   
    lastX, lastY = 800 / 2, 600 / 2  # Initially set to the center of the screen
    fov = 90.0

 

    projection = pyrr.Matrix44.perspective_projection(fov, 2, 0.1, 1000.0)
    def process_input(window, delta_time):
        global player_pos, player_front, camera_speed

        camera_speed = 15.0 * delta_time
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            player_pos += camera_speed * player_front
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            player_pos -= camera_speed * player_front
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            player_pos -= Vector3.cross(player_front, camera_up).normalize() * camera_speed
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            player_pos += Vector3.cross(player_front, camera_up).normalize() * camera_speed

    def mouse_callback(window, xpos, ypos):
        global lastX, lastY, yaw, pitch, player_front

        xoffset = xpos - lastX
        yoffset = lastY - ypos  # Reversed since y-coordinates go from bottom to top
        lastX = xpos
        lastY = ypos

        sensitivity = 0.1
        xoffset *= sensitivity
        yoffset *= sensitivity

        yaw += xoffset
        pitch += yoffset

        if pitch > 89.0:
            pitch = 89.0
        if pitch < -89.0:
            pitch = -89.0

        front = Vector3([0.0, 0.0, 0.0])
        front.x = np.cos(np.radians(yaw)) * np.cos(np.radians(pitch))
        front.y = np.sin(np.radians(pitch))
        front.z = np.sin(np.radians(yaw)) * np.cos(np.radians(pitch))
        player_front = front.normalize()
    glfw.set_cursor_pos_callback(window, mouse_callback)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

   

    
    
    def create_view_matrix(player_pos, target, up):
        return glm.lookAt(player_pos, target, up)

    
    def create_projection_matrix(fov, aspect_ratio, near_plane, far_plane):
        return glm.perspective(glm.radians(fov), aspect_ratio, near_plane, far_plane)

    # Example usage
    player_pos = glm.vec3(0, 20, -10)  # Camera position
    camera_target = glm.vec3(0, 0, 0)  # Where the camera is looking
    camera_up = glm.vec3(0, 1, 0)  # Up direction for the camera

    view_matrix = create_view_matrix(player_pos, camera_target, camera_up)
    projection_matrix = create_projection_matrix(45, 2, 0.1, 5000.0)


    # Make the window's context current
    glfw.make_context_current(window)

    # glEnable(GL_BLEND)
    # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


    # Compile shaders and create shader program
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex_shader, vertex_shader_source)
    glCompileShader(vertex_shader)
    if not glGetShaderiv(vertex_shader, GL_COMPILE_STATUS):
        print("Vertex shader compilation error: ", glGetShaderInfoLog(vertex_shader))
        return

    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragment_shader, fragment_shader_source)
    glCompileShader(fragment_shader)
    if not glGetShaderiv(fragment_shader, GL_COMPILE_STATUS):
        print("Fragment shader compilation error: ", glGetShaderInfoLog(fragment_shader))
        return

    shader_program = glCreateProgram()

    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)

    if not glGetProgramiv(shader_program, GL_LINK_STATUS):
        print("Shader link error: ", glGetProgramInfoLog(shader_program))
        return
    glUseProgram(shader_program)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, load_texture('dirt.png'))
    glUniform1i(glGetUniformLocation(shader_program, "texture1"), 0)
    
    rot = 0
    
    chunks = []
    global quad_count, VAO, VBO

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    global vertex_data
    chunk_data_size = 16 * 32 * 16 * 6 * 4 * 3
    max_chunks = 256
    world_width = int(np.sqrt(max_chunks))
    chunks = []
    # for x in range(world_width):
    #     for y in range(world_width):
    #         chunks.append(generate_chunk((x - int(world_width/2), y - int(world_width/2), x * world_width + y)))
      
    #         print(np.round(((x * world_width + y) / max_chunks * 100), 3), "%")
         
    
 
    glEnable(GL_DEPTH_TEST)
    last_frame = glfw.get_time()

     # Generate and bind a Vertex Array Object (VAO)
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)


    # Generate and bind a Vertex Buffer Object (VBO)
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    # Render loop

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * 4, ctypes.c_void_p(0))

    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 9 * 4, ctypes.c_void_p(3 * 4))
    # Texture coordinate attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 9 * 4, ctypes.c_void_p(7 * 4))
    glEnableVertexAttribArray(2)
    glEnableVertexAttribArray(1)
    glEnableVertexAttribArray(0)

    glUseProgram(shader_program)
    glBindVertexArray(VAO)
    
    quad_count = 0
   
    future_chunks = []
    index_counter = 0
    pending_chunk_results = []
    vertex_data = np.empty(chunk_data_size * max_chunks * 2, dtype=np.float32)
    glBufferData(GL_ARRAY_BUFFER, len(vertex_data) * 4, None, GL_STATIC_DRAW)
    while not glfw.window_should_close(window):
        
        def render(_chunks):
            global quad_count, VAO, VBO, vertex_data
            
            def flatten_concatenation(matrix):
                flat_list = []
                for row in matrix:
                    if row != None:
                        flat_list += row
                return flat_list
            updated = False
            for grid in _chunks:
                
                if grid.updated == True:
                    continue
                grid.updated = True
                updated = True

                north = None
                south = None
                east = None
                west = None

                for c in chunks:
                    if c.x == grid.x + 1:
                        east = c
                    if c.x == grid.x - 1:
                        west = c
                    if c.y == grid.y + 1:
                        north = c
                    if c.y == grid.y - 1:
                        south = c
                
                
                
                adjacent = (north, south, east, west)
            
                #print("generating chunk")
                grid_vertex_data = [] 
                grid_transparent_data = []
                for x in range(len(grid.blocks)):
                    for y in range(len(grid.blocks[x])):
                        for z in range(len(grid.blocks[x][y])):

                            block = grid.blocks[x][y][z]
                            block.omit_faces(grid, x, y, z, adjacent)
                            if block.id == 1 or block.id == 3 or block.id == 4:
                                result = draw_cube(Vector3(x*2 + grid.x * 32, y*2, z*2 + grid.y * 32), rot, block.omitables, block.id)
                                quad_count += len(result) / 3
                                grid_vertex_data.append(result)
                
                for x in range(len(grid.blocks)):
                    for y in range(len(grid.blocks[x])):
                        for z in range(len(grid.blocks[x][y])):

                            block = grid.blocks[x][y][z]
                            block.omit_faces(grid, x, y, z, adjacent)
                            if block.id == 2:
                                result = draw_cube(Vector3(x*2 + grid.x * 32, y*2, z*2 + grid.y * 32), rot, block.omitables, block.id)
                                quad_count += len(result) / 3
                                grid_transparent_data.append(result)
            
                grid_vertex_data = flatten_concatenation(grid_vertex_data)
              
                grid_transparent_data = flatten_concatenation(grid_transparent_data)
                grid_vertex_data = np.array(grid_vertex_data, dtype=np.float32)
                grid_transparent_data = np.array(grid_transparent_data, dtype=np.float32)
            #     vertex_data[grid.index * chunk_data_size:grid.index * chunk_data_size + len(grid_vertex_data)] = grid_vertex_data
            #     vertex_data[grid.index * chunk_data_size + chunk_data_size * max_chunks:grid.index * chunk_data_size + len(grid_transparent_data) + chunk_data_size * max_chunks] = grid_transparent_data
            # # if updated:
                glBufferSubData(GL_ARRAY_BUFFER, (grid.index * chunk_data_size) * 4, grid_vertex_data.nbytes, grid_vertex_data)
                glBufferSubData(GL_ARRAY_BUFFER, (grid.index * chunk_data_size + chunk_data_size * max_chunks) * 4, grid_transparent_data.nbytes, grid_transparent_data)

                    # Link vertex attributes
            #if updated:
     
    
        
                    
        
        
        current_frame = glfw.get_time()
        delta_time = current_frame - last_frame
        last_frame = current_frame
        process_input(window, delta_time)  # Assuming delta_time is calculated
    
        # Update view matrix based on camera's updated position and direction
        view = pyrr.Matrix44.look_at(player_pos, player_pos + player_front, camera_up).astype(np.float32)
        
        # Clear the screen
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        
        projection_loc = glGetUniformLocation(shader_program, "projection")
        glUniformMatrix4fv(projection_loc, 1, GL_FALSE, projection.astype(np.float32))

        chunk_pos = Vector2(int(np.floor(player_pos.x / 32)), int(np.floor(player_pos.z / 32)))
        #print(chunk_pos)
        chunks_to_load = [(chunk_pos.x + x, chunk_pos.y + y) for x in range(-1, 2) for y in range(-1, 2) if all(c.x != chunk_pos.x + x or c.y != chunk_pos.y + y for c in chunks)]

        # Submit new chunk generation tasks
    
        
                
     
            #future = executor.submit(generate_chunk, pos[0], pos[1], noise, len(chunks))
            # process = mp.Process(target=generate_chunk, args=(pos[0], pos[1], noise, len(chunks)))
            # process.start()
           
                #future_chunks.append(future)
     
        #return_dict = manager.dict()
        # jobs = []
        # result_queue = mp.Queue()  # Queue for collecting results
        # total_chunks = len(chunks_to_load)  # Assuming this is what you meant with len(chunks)
        
        # for pos in chunks_to_load:
        #     p = mp.Process(target=generate_chunk, args=(pos[0], pos[1], noise, total_chunks, result_queue))
        #     jobs.append(p)
        #     p.start()
        # print("joining jobs")
        # for job in jobs:
        #     job.join()  # Wait for all processes to complete

      
        # while not result_queue.empty():
        #     chunks.append(result_queue.get())

      
         
        for pos in chunks_to_load:
            if len(chunks) < max_chunks:
                chunks.append(generate_chunk((pos[0], pos[1], len(chunks))))
            # else:
            #     chunks.pop(0)
       
                   

        # # Now check for and process any completed tasks
        # new_chunks = []
        # for res in pending_chunk_results:
           
        #     if res.ready():
        #         new_chunks.append(res.get())
        #         pending_chunk_results.remove(res)
        
        # if new_chunks:
         
        #     chunks.extend(new_chunks)
            # Any necessary updates after adding new chunks
            # For example, updating OpenGL buffers for rendering

        if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
            grid = None
            for c in chunks:
                if chunk_pos.x == c.x and chunk_pos.y == c.y:
                    grid = c
                    break
            
            if grid != None:
                xpos = int((player_pos.x / 2 + player_front.x * 2)% grid.w)
                zpos = int((player_pos.z / 2 + player_front.z * 2)% grid.d)
                ypos = int(player_pos.y/2 + player_front.y * 2)
                if ypos > grid.h - 1:
                    ypos = grid.h - 1
                if ypos < 0:
                    ypos = 0
                if grid.blocks[xpos][ypos][zpos].id == 0:
                    grid.blocks[xpos][ypos][zpos] = Block(3)
                    grid.updated = False   
        elif glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:   
            grid = None
            for c in chunks:
                if chunk_pos.x == c.x and chunk_pos.y == c.y:
                    grid = c
                    break
            
            if grid != None:
                xpos = int((player_pos.x / 2 + player_front.x * 2)% grid.w)
                zpos = int((player_pos.z / 2 + player_front.z * 2)% grid.d)
                ypos = int(player_pos.y/2 + player_front.y * 2)
                if ypos > grid.h - 1:
                    ypos = grid.h - 1
                if ypos < 0:
                    ypos = 0
                if grid.blocks[xpos][ypos][zpos].id != 0:
                    grid.blocks[xpos][ypos][zpos] = Block(0)
                    grid.updated = False        

        render(chunks)
        
        # can_spawn = True
        # for c in chunks:
        #     if c.x == chunk_pos.x and c.y == chunk_pos.y:
        #         can_spawn = False
        # if can_spawn:
        #     chunks.append(generate_chunk(chunk_pos.x, chunk_pos.y, noise, index_counter))
            
        #     print(chunk_pos.x, chunk_pos.y)
        #     
        
        # # Collect completed tasks and update chunks
        # completed_futures = [f for f in future_chunks if f.done()]
        # for future in completed_futures:
        #     chunks.append(future.result())
        
        #     future_chunks.remove(future)

        # for i in range(len(chunks), 0, -1):
        #     c = chunks[i - 1]
        #     for d in chunks:
        #         if c != d:
        #             if c.x == d.x and c.y == d.y:
        #                 chunks.remove(c)
        
        
    


        
        view_loc = glGetUniformLocation(shader_program, "view")
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        
        glDrawArrays(GL_QUADS, 0, int(6 * quad_count))

        # Swap front and back buffers
        glfw.swap_buffers(window)

        # Poll for and process events
        glfw.poll_events()

    # Clean up
    glfw.terminate()

if __name__ == "__main__":
    main()
