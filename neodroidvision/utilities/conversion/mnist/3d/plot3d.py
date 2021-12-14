import numpy
from IPython.display import IFrame
from matplotlib import pyplot


def array_to_color(array, cmap="Oranges") -> object:
  """

  Args:
    array:
    cmap:

  Returns:

  """
  s_m = pyplot.cm.ScalarMappable(cmap=cmap)
  return s_m.to_rgba(array)[:, :-1]


TEMPLATE_POINTS = """
<!DOCTYPE html>
<head>

<title>PyntCloud</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, user-scalable=no, minimum-scale=1.0, maximum-scale=1.0">
<style>
body {{
	color: #cccccc;
	font-family: Monospace;
	font-size: 13px;
	text-align: center;
	background-color: #050505;
	margin: 0px;
	overflow: hidden;
}}
#logo_container {{
	position: absolute;
	top: 0px;
	width: 100%;
}}
.logo {{
	max-width: 20%;
}}
</style>

</head>
<body>

<div>
	<img class="logo" src="https://media.githubusercontent.com/media/daavoo/pyntcloud/master/docs/data/pyntcloud.png">
</div>

<div id="container">
</div>

<script src="http://threejs.org/build/three.js"></script>
<script src="http://threejs.org/examples/js/Detector.js"></script>
<script src="http://threejs.org/examples/js/controls/OrbitControls.js"></script>
<script src="http://threejs.org/examples/js/libs/stats.min.js"></script>

<script>

	if ( ! Detector.webgl ) Detector.addGetWebGLMessage();

	var container, stats;
	var camera, scene, renderer;
	var points;

	init();
	animate();

	function init() {{

		var camera_x = {camera_x};
		var camera_y = {camera_y};
		var camera_z = {camera_z};

        var look_x = {look_x};
        var look_y = {look_y};
        var look_z = {look_z};

		var positions = new Float32Array({positions});

		var colors = new Float32Array({colors});

		var points_size = {points_size};

		var axis_size = {axis_size};

		container = document.getElementById( 'container' );

		scene = new THREE.Scene();

		camera = new THREE.PerspectiveCamera( 90, window.innerWidth / window.innerHeight, 0.1, 1000 );
		camera.position.x = camera_x;
		camera.position.y = camera_y;
		camera.position.z = camera_z;
		camera.up = new THREE.Vector3( 0, 0, 1 );		

		if (axis_size > 0){{
            var axisHelper = new THREE.AxisHelper( axis_size );
		    scene.add( axisHelper );
        }}

		var geometry = new THREE.BufferGeometry();
		geometry.addAttribute( 'position', new THREE.BufferAttribute( positions, 3 ) );
		geometry.addAttribute( 'color', new THREE.BufferAttribute( colors, 3 ) );
		geometry.computeBoundingSphere();

		var material = new THREE.PointsMaterial( {{ size: points_size, vertexColors: THREE.VertexColors }} );

		points = new THREE.Points( geometry, material );
		scene.add( points );


		renderer = new THREE.WebGLRenderer( {{ antialias: false }} );
		renderer.setPixelRatio( window.devicePixelRatio );
		renderer.setSize( window.innerWidth, window.innerHeight );

		controls = new THREE.OrbitControls( camera, renderer.domElement );
		controls.target.copy( new THREE.Vector3(look_x, look_y, look_z) );
        camera.lookAt( new THREE.Vector3(look_x, look_y, look_z));

		container.appendChild( renderer.domElement );

		window.addEventListener( 'resize', onWindowResize, false );
	}}

	function onWindowResize() {{
		camera.aspect = window.innerWidth / window.innerHeight;
		camera.updateProjectionMatrix();
		renderer.setSize( window.innerWidth, window.innerHeight );
	}}

	function animate() {{
		requestAnimationFrame( animate );
		render();
	}}

	function render() {{
		renderer.render( scene, camera );
	}}
</script>

</body>
</html>
"""


def plot_points(xyz, colors=None, size=0.1, axis=False):
  """

  Args:
    xyz:
    colors:
    size:
    axis:

  Returns:

  """
  positions = xyz.reshape(-1).tolist()

  camera_position = xyz.max(0) + abs(xyz.max(0))

  look = xyz.mean(0)

  if colors is None:
    colors = [1, 0.5, 0] * len(positions)

  elif len(colors.shape) > 1:
    colors = colors.reshape(-1).tolist()

  if axis:
    axis_size = xyz.ptp() * 1.5
  else:
    axis_size = 0

  with open("plot_points.html", "w") as html:
    html.write(TEMPLATE_POINTS.format(
        camera_x=camera_position[0],
        camera_y=camera_position[1],
        camera_z=camera_position[2],
        look_x=look[0],
        look_y=look[1],
        look_z=look[2],
        positions=positions,
        colors=colors,
        points_size=size,
        axis_size=axis_size))

  return IFrame("plot_points.html", width=800, height=800)


TEMPLATE_VG = """
<!DOCTYPE html>
<head>

<title>PyntCloud</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, user-scalable=no, minimum-scale=1.0, maximum-scale=1.0">
<style>
    body {{
        color: #cccccc;font-family: Monospace;
        font-size: 13px;
        text-align: center;
        background-color: #050505;
        margin: 0px;
        overflow: hidden;
    }}
    #logo_container {{
        position: absolute;
        top: 0px;
        width: 100%;
    }}
    .logo {{
        max-width: 20%;
    }}
</style>

</head>
<body>

<div>
    <img class="logo" src="https://media.githubusercontent.com/media/daavoo/pyntcloud/master/docs/data/pyntcloud.png">
</div>

<div id="container">
</div>

<script src="http://threejs.org/build/three.js"></script>
<script src="http://threejs.org/examples/js/Detector.js"></script>
<script src="http://threejs.org/examples/js/controls/OrbitControls.js"></script>
<script src="http://threejs.org/examples/js/libs/stats.min.js"></script>

<script>

    if ( ! Detector.webgl ) Detector.addGetWebGLMessage();

    var container, stats;
    var camera, scene, renderer;
    var points;

    init();
    animate();

    function init() {{

        var camera_x = {camera_x};
		var camera_y = {camera_y};
		var camera_z = {camera_z};

        var look_x = {look_x};
        var look_y = {look_y};
        var look_z = {look_z};

		var X = new Float32Array({X});
        var Y = new Float32Array({Y});
        var Z = new Float32Array({Z});

        var R = new Float32Array({R});
        var G = new Float32Array({G});
        var B = new Float32Array({B});

        var S_x = {S_x};
        var S_y = {S_y};
        var S_z = {S_z};

        var n_voxels = {n_voxels};
        var axis_size = {axis_size};

        container = document.getElementById( 'container' );

        scene = new THREE.Scene();

        camera = new THREE.PerspectiveCamera( 90, window.innerWidth / window.innerHeight, 0.1, 1000 );
        camera.position.x = camera_x;
        camera.position.y = camera_y;
        camera.position.z = camera_z;
        camera.up = new THREE.Vector3( 0, 0, 1 );	

        if (axis_size > 0){{
            var axisHelper = new THREE.AxisHelper( axis_size );
		    scene.add( axisHelper );
        }}

        var geometry = new THREE.BoxGeometry( S_x, S_z, S_y );

        for ( var i = 0; i < n_voxels; i ++ ) {{            
            var mesh = new THREE.Mesh( geometry, new THREE.MeshBasicMaterial() );
            mesh.material.color.setRGB(R[i], G[i], B[i]);
            mesh.position.x = X[i];
            mesh.position.y = Y[i];
            mesh.position.z = Z[i];
            scene.add(mesh);
        }}

        renderer = new THREE.WebGLRenderer( {{ antialias: false }} );
        renderer.setPixelRatio( window.devicePixelRatio );
        renderer.setSize( window.innerWidth, window.innerHeight );

        controls = new THREE.OrbitControls( camera, renderer.domElement );
        controls.target.copy( new THREE.Vector3(look_x, look_y, look_z) );
        camera.lookAt( new THREE.Vector3(look_x, look_y, look_z));

        container.appendChild( renderer.domElement );

        window.addEventListener( 'resize', onWindowResize, false );
    }}

    function onWindowResize() {{
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize( window.innerWidth, window.innerHeight );
    }}

    function animate() {{
        requestAnimationFrame( animate );
        render();
    }}

    function render() {{
        renderer.render( scene, camera );
    }}
</script>
</body>
</html>
"""


def plot_voxelgrid(v_grid, cmap="Oranges", axis=False):
  """

  Args:
    v_grid:
    cmap:
    axis:

  Returns:

  """
  scaled_shape = v_grid.shape / min(v_grid.shape)

  # coordinates returned from argwhere are inversed so use [:, ::-1]
  points = numpy.argwhere(v_grid.vector)[:, ::-1] * scaled_shape

  s_m = pyplot.cm.ScalarMappable(cmap=cmap)
  rgb = s_m.to_rgba(v_grid.vector.reshape(-1)[v_grid.vector.reshape(-1) > 0])[:, :-1]

  camera_position = points.max(0) + abs(points.max(0))
  look = points.mean(0)

  if axis:
    axis_size = points.ptp() * 1.5
  else:
    axis_size = 0

  with open("plotVG.html", "w") as html:
    html.write(TEMPLATE_VG.format(
        camera_x=camera_position[0],
        camera_y=camera_position[1],
        camera_z=camera_position[2],
        look_x=look[0],
        look_y=look[1],
        look_z=look[2],
        X=points[:, 0].tolist(),
        Y=points[:, 1].tolist(),
        Z=points[:, 2].tolist(),
        R=rgb[:, 0].tolist(),
        G=rgb[:, 1].tolist(),
        B=rgb[:, 2].tolist(),
        S_x=scaled_shape[0],
        S_y=scaled_shape[2],
        S_z=scaled_shape[1],
        n_voxels=sum(v_grid.vector.reshape(-1) > 0),
        axis_size=axis_size))

  return IFrame("plotVG.html", width=800, height=800)


class VoxelGrid(object):
  """

  """
  def __init__(self, points, x_y_z=(1, 1, 1), bb_cuboid=True, build=True):
    """
    Parameters
    ----------
    points: (N,3) ndarray
            The point cloud from wich we want to construct the VoxelGrid.
            Where N is the number of points in the point cloud and the second
            dimension represents the x, y and z coordinates of each point.

    x_y_z:  list
            The segments in wich each axis will be divided.
            x_y_z[0]: x axis
            x_y_z[1]: y axis
            x_y_z[2]: z axis

    bb_cuboid(Optional): bool
            If True(Default):
                The bounding box of the point cloud will be adjusted
                in order to have all the dimensions of equal lenght.
            If False:
                The bounding box is allowed to have dimensions of different sizes.
    """
    self.points = points

    xyzmin = numpy.min(points, axis=0) - 0.001
    xyzmax = numpy.max(points, axis=0) + 0.001

    if bb_cuboid:
      #: adjust to obtain a  minimum bounding box with all sides of equal lenght
      diff = max(xyzmax - xyzmin) - (xyzmax - xyzmin)
      xyzmin = xyzmin - diff / 2
      xyzmax = xyzmax + diff / 2

    self.xyzmin = xyzmin
    self.xyzmax = xyzmax

    segments = []
    shape = []

    for i in range(3):
      # note the +1 in num
      if type(x_y_z[i]) is not int:
        raise TypeError(f"x_y_z[{i}] must be int")
      s, step = numpy.linspace(xyzmin[i], xyzmax[i], num=(x_y_z[i] + 1), retstep=True)
      segments.append(s)
      shape.append(step)

    self.segments = segments

    self.shape = shape

    self.n_voxels = x_y_z[0] * x_y_z[1] * x_y_z[2]
    self.n_x = x_y_z[0]
    self.n_y = x_y_z[1]
    self.n_z = x_y_z[2]

    self.id = f"{x_y_z[0]},{x_y_z[1]},{x_y_z[2]}-{bb_cuboid}"

    if build:
      self.build()

  def build(self):
    """

    """
    structure = numpy.zeros((len(self.points), 4), dtype=int)

    structure[:, 0] = numpy.searchsorted(self.segments[0], self.points[:, 0]) - 1

    structure[:, 1] = numpy.searchsorted(self.segments[1], self.points[:, 1]) - 1

    structure[:, 2] = numpy.searchsorted(self.segments[2], self.points[:, 2]) - 1

    # i = ((y * n_x) + x) + (z * (n_x * n_y))
    structure[:, 3] = ((structure[:, 1] * self.n_x) + structure[:, 0]) + (structure[:, 2] * (self.n_x * self.n_y))

    self.structure = structure

    vector = numpy.zeros(self.n_voxels)
    count = numpy.bincount(self.structure[:, 3])
    vector[:len(count)] = count

    self.vector = vector.reshape(self.n_z, self.n_y, self.n_x)

  def plot(self, d=2, cmap="Oranges", axis=False):
    """

    Args:
      d:
      cmap:
      axis:

    Returns:

    """
    if d == 2:

      fig, axes = pyplot.subplots(int(numpy.ceil(self.n_z / 4)), 4, figsize=(8, 8))

      pyplot.tight_layout()

      for i, ax in enumerate(axes.flat):
        if i >= len(self.vector):
          break
        im = ax.imshow(self.vector[i], cmap=cmap, interpolation="none")
        ax.set_title("Level " + str(i))

      fig.subplots_adjust(right=0.8)
      cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
      cbar = fig.colorbar(im, cax=cbar_ax)
      cbar.set_label('NUMBER OF POINTS IN VOXEL')

    elif d == 3:
      return plot_voxelgrid(self, cmap=cmap, axis=axis)
