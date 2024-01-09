import bpy
import bmesh
import math

def update_sine_graph(scene):
    # Clear existing mesh objects in the scene
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    # Create a new mesh object
    mesh = bpy.data.meshes.new(name="SineGraph")
    obj = bpy.data.objects.new("SineGraph", mesh)

    # Link the new object to the scene
    scene.collection.objects.link(obj)
    scene.objects.active = obj
    obj.select_set(True)

    # Create a BMesh to define the mesh geometry
    bm = bmesh.new()

    # Parameters for the sine wave
    amplitude = 1.0
    frequency = 2.0
    num_vertices = 100

    for i in range(num_vertices):
        x = i * 0.1  # Adjust the spacing as needed
        y = amplitude * math.sin(frequency * x)
        z = 0.0
        bm.verts.new((x, y, z))

    # Create edges connecting vertices
    bm.edges.new(bm.verts)

    # Update the mesh with the BMesh data
    bm.to_mesh(mesh)
    bm.free()

    # Set up the materials for the mesh
    mat = bpy.data.materials.new(name="SineMaterial")
    mat.diffuse_color = (0.0, 0.0, 1.0, 1.0)
    mesh.materials.append(mat)

# Register the update function with the frame change event
bpy.app.handlers.frame_change_pre.append(update_sine_graph)

# Run Blender in the background
bpy.ops.wm.console_toggle()
