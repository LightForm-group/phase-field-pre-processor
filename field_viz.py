from subprocess import call
from numpy.lib.arraysetops import isin
from plotly import graph_objects
import ipywidgets
import numpy as np

from quats import quat_angle_between


def get_cart_vector(vec_str):
    """Convert a string {+/-}{xyz} into a Cartesian unit vector."""
    vec = [0, 0, 0]
    vec['xyz'.index(vec_str[1])] = 1 * -1 if vec_str[0] == '-' else 1
    return np.array(vec)


def get_volumetric_line_profile(arr, x=None, y=None, z=None, mask=None):
    """Get a line of data through a 3D array.

    Parameters
    ----------
    arr : ndarray of shape (N, M, P, ...)
        Field data array where N, M, P correspond to the grid size in x, y, z,
        respectively.
    x : int, optional
    y : int, optional
    z : int, optional
    mask : ndarray of same shape as `arr` of bool

    Notes
    -----
    Two of "x", "y" and "z" must be specified. The unspecified argument represents the
    line direction.

    Returns
    -------
    line : ndarray of shape (i, ...) where i is one of {N, M, P}

    """
    if sum([i is None for i in (x, y, z)]) != 1:
        raise ValueError('Specify exactly two of "x", "y", and "z".')

    x_slice = slice(None) if x is None else x
    y_slice = slice(None) if y is None else y
    z_slice = slice(None) if z is None else z

    line = arr[x_slice, y_slice, z_slice]

    if mask is not None:
        line_profile_mask = get_volumetric_line_profile(mask, x=x, y=y, z=z)
        line[line_profile_mask] = np.nan

    return line


def validate_vol_slice_parametrisation(x, y, z, eye, up):

    if sum([i is None for i in (x, y, z)]) != 2:
        raise ValueError('Specify exactly one of "x", "y", and "z".')

    normal_dir = 'x' if x is not None else ('y' if y is not None else 'z')
    slice_idx = x if x is not None else (y if y is not None else z)

    IMPLICIT_DIRS = {  # keys are plane normals, values are implicit "defaults" from Numpy slicing:
        'x': {
            'eye': '-x',
            'up': '-y',
        },
        'y': {
            'eye': '+y',
            'up': '-x',
        },
        'z': {
            'eye': '-z',
            'up': '-x',
        },
    }

    ALLOWED_DIRS = {  # keys are plane normals
        i: {
            'eye': (f'+{i}', f'-{i}'),
            'up': tuple(f'{k}{j}' for j in set('xyz') - {i} for k in ('-', '+')),
        }
        for i in 'xyz'
    }

    DEFAULT_UP_DIRS = {  # keys are eye dirs, defaults to match ParaView preselect buttons
        '+x': '+z',
        '-x': '+z',
        '+y': '+z',
        '-y': '+z',
        '+z': '+y',
        '-z': '+y',
    }

    if eye:
        if len(eye) == 1:
            eye = f'+{eye}'
        eye = eye.lower()

    if up:
        if len(up) == 1:
            up = f'+{up}'
        up = up.lower()

    if eye is None:
        eye = f'+{normal_dir}'
        print(f'eye is not specified, setting to: {eye}')

    if up is None:
        up = DEFAULT_UP_DIRS[eye]

    allowed_eyes = ALLOWED_DIRS[normal_dir]['eye']
    if eye not in allowed_eyes:
        msg = f'`eye` must be one of: {allowed_eyes}, but was specified as: {eye}.'
        raise ValueError(msg)

    allowed_ups = ALLOWED_DIRS[normal_dir]['up']
    if up not in allowed_ups:
        msg = f'`up` must be one of: {allowed_ups}, but was specified as: {up}.'
        raise ValueError(msg)

    slices = [slice(None), slice(None), slice(None)]
    slices['xyz'.index(normal_dir)] = slice_idx

    # keys are plane normals, vals are resulting up dir from a sequence of anti-clockwise
    # array rotations:
    UP_ROT_SEQUENCE = {
        'x': ('-y', '+z', '+y', '-z'),
        'y': ('-x', '+z', '+x', '-z'),
        'z': ('-x', '+y', '+x', '-y'),
    }

    out = {
        'x_slice': slices[0],
        'y_slice': slices[1],
        'z_slice': slices[2],
        'eye': eye,
        'up': up,
        'num_up_rotations': UP_ROT_SEQUENCE[normal_dir].index(up),
        'flip_horizontal': eye != IMPLICIT_DIRS[normal_dir]['eye'],
    }

    return out


def get_volumetric_slice(arr, x=None, y=None, z=None, eye=None, up=None, mask=None,
                         transpose=False):
    """Get a slice of data through a 3D array.

    Parameters
    ----------
    arr : ndarray of shape (N, M, P, ...)
        Field data array where N, M, P correspond to the grid size in x, y, z,
        respectively.
    x : int, optional
    y : int, optional
    z : int, optional

    Notes
    -----
    One of "x", "y" and "z" must be specified. The specified arguments represents the
    slice normal direction.

    Returns
    -------
    slice_data : ndarray of shape (i, j, ...) where {i,j} are each one of {N, M, P}

    """

    valid_param = validate_vol_slice_parametrisation(x, y, z, eye, up)

    x_slice = valid_param['x_slice']
    y_slice = valid_param['y_slice']
    z_slice = valid_param['z_slice']

    slice_data = arr[x_slice, y_slice, z_slice]

    if mask is not None:
        slice_data_mask = get_volumetric_slice(mask, x=x, y=y, z=z)['slice_data']
        slice_data[slice_data_mask] = np.nan

    slice_data = np.rot90(slice_data, k=valid_param['num_up_rotations'])
    if valid_param['flip_horizontal']:
        slice_data = np.fliplr(slice_data)

    y_lab = valid_param['up']

    eye_vec = get_cart_vector(valid_param['eye'])
    up_vec = get_cart_vector(valid_param['up'])

    x_lab_vec = np.cross(eye_vec, up_vec)
    x_lab_dir = 'xyz'[np.where(np.abs(x_lab_vec) == 1)[0][0]]
    x_lab_sign = '-' if np.sum(x_lab_vec) < 0 else '+'

    x_lab = f'{x_lab_sign}{x_lab_dir}'

    if transpose:
        slice_data = slice_data.T
        x_lab, y_lab = y_lab, x_lab

    out = {
        'slice_data': slice_data,
        'xlabel': x_lab,
        'ylabel': y_lab,
    }

    return out


def get_misorientation_line_profile(quats, x=None, y=None, z=None, mask=None):
    """Get the angular misorientation in degrees between the first quaternion and all
    subsequent quaternions along a line profile within the 3D volume element.        
    """

    line_profile_quats = get_volumetric_line_profile(quats, x=x, y=y, z=z, mask=mask)
    ref_quat = line_profile_quats[0]
    ref_quat_tiled = np.tile(ref_quat, (line_profile_quats.shape[0] - 1, 1))
    misori = np.concatenate((
        [0],
        np.rad2deg(quat_angle_between(ref_quat_tiled, line_profile_quats[1:])),
    ))

    return misori


def show_misorientation_line_profile(quats, x=None, y=None, z=None, mask=None):
    misori = get_misorientation_line_profile(quats, x=x, y=y, z=z, mask=mask)
    layout = {
        'xaxis': {
            'title': 'Line profile position',
        },
        'yaxis': {
            'title': 'Misorientation / deg.'
        }
    }
    data = [
        {
            'y': misori,
        }
    ]
    fig = graph_objects.FigureWidget(
        data=data,
        layout=layout,
    )
    return fig


def show_volumetric_slice(arr, x=None, y=None, z=None, eye=None, up=None, mask=None,
                          transpose=False, data_name=''):

    slice_data = get_volumetric_slice(arr, x=x, y=y, z=z, eye=eye, up=up, mask=mask,
                                      transpose=transpose)

    fig = graph_objects.FigureWidget(
        data=[
            {
                'type': 'heatmap',
                # plotly heatmap origin is bottom left, rather than top left:
                'z': slice_data['slice_data'][::-1],
                'colorscale': 'Viridis',
                # 'zmin': 0,
                # 'zmax': arr.max(),
                'colorbar': {
                    # 'tickformat': '.2f',
                    'title': data_name,
                }
            },
        ],
        layout={
            'template': 'none',
            'margin': {
                't': 20,
                'r': 50,
                'b': 50,
                'l': 50,
            },
            'height': 500,
            'width': 700,
            'xaxis': {
                'scaleanchor': 'y',
                'constrain': 'domain',
                'title': {'text': slice_data['xlabel'], 'font': {'size': 22}},
                'showgrid': False,
                'showticklabels': False,
                'zeroline': False,
                'ticks': '',
            },
            'yaxis': {
                'title': {'text': slice_data['ylabel'], 'font': {'size': 22}},
                'showgrid': False,
                'showticklabels': False,
                'ticks': '',
                'zeroline': False,
            }
        }
    )
    return fig


def get_plotly_discrete_colour_bar(discrete_values, colors):
    """Get a dict with `colorbar` and `colorscale` keys in for a discrete color bar."""

    color_scale_inc = 1 / len(discrete_values)
    color_scale = [
        [
            [segment * color_scale_inc, colors[segment % len(colors)]],
            [(segment * color_scale_inc) + color_scale_inc, colors[segment % len(colors)]],
        ]
        for segment in range(len(discrete_values))
    ]

    color_scale = [j for i in color_scale for j in i]  # flatten
    color_bar = [(segment * color_scale_inc) + (color_scale_inc / 2)
                 for segment in range(len(discrete_values))]

    out = {
        'colorscale': color_scale,
        'colorbar': {
            'tickmode': 'array',
            'tickvals': color_bar,
            'ticktext': discrete_values,
        },
    }

    return out


class RVEFieldViz:

    def __init__(self, volume_element_response, increment=None, x=None, y=None, z=None,
                 eye=None, up=None, transpose=False, data_name=None, data_component=None,
                 line_data_horizontal_name=None, line_data_vertical_name=None,
                 global_colour_scale=True, callbacks=None):

        self.volume_element_response = volume_element_response
        self.increment_idx = increment if increment is not None else -1
        self.x = x
        self.y = y
        self.z = z
        self.eye = eye
        self.up = up
        self.transpose = transpose
        self.data_name = data_name
        self.data_component = data_component
        self.global_colour_scale = global_colour_scale
        self.callbacks = callbacks or {}
        self.current_callback = 'None'
        self.line_hor_position = 0
        self.line_ver_position = 0

        self._validate()

        self._set_plot_data(
            self.data_name,
            self.data_component,
            self.increment_idx,
            self.current_callback,
        )

        self.figure = self._generate_figure()
        self._widgets = self._generate_widgets()

    def _validate(self):
        """Check valid data names and set defaults."""
        if self.data_name:
            if self.data_name not in self.field_data_names:
                raise ValueError(f'Data name must be a valid field data, one '
                                 f'of: {self.field_data_names}')
        else:
            self.data_name = self._first_field_data_name

        _ = validate_vol_slice_parametrisation(self.x, self.y, self.z, self.eye, self.up)

        if self.data_component is not None:
            if isinstance(self.data_component, int):
                self.data_component = (self.data_component,)
            elif isinstance(self.data_component, list):
                self.data_component = tuple(self.data_component)
        else:
            self.data_component = self._get_allowed_tensorial_components(self.data_name)[
                0]

        if (self.data_component not in
                self._get_allowed_tensorial_components(self.data_name)):
            msg = (f'Data "{self.data_name}" has inner (tensorial) shape '
                   f'{self._get_field_data_inner_shape(self.data_name)}, which is '
                   f'incompatible with the specified plane data component: '
                   f'{self.data_component}')
            raise ValueError(msg)

        if self.increment_idx < 0:
            self.increment_idx += self.get_num_increments(self.data_name)

    def _set_plot_data(self, data_name, data_component, increment, callback):

        data_arr_all_incs = self.get_field_data(data_name)
        data_arr = data_arr_all_incs[increment]

        # print(f'data_arr_all_incs.shape: {data_arr_all_incs.shape}')
        # print(f'data_arr.shape: {data_arr.shape}')

        plane_data = get_volumetric_slice(
            data_arr,
            x=self.x,
            y=self.y,
            z=self.z,
            eye=self.eye,
            up=self.up,
            # mask=mask,
            transpose=self.transpose,
        )
        plane_data_arr = plane_data['slice_data']

        # print(f'data_component: {data_component}')

        if callback == 'None':
            nd_slice = [slice(None) for _ in range(plane_data_arr.ndim)]
            nd_slice[-len(data_component):] = data_component
            self.plot_data = plane_data_arr[tuple(nd_slice)]
            title = f'{data_name}[{data_component}]' if data_component else data_name

        else:
            self.plot_data = self.callbacks[callback](plane_data_arr)
            title = f'{callback}({data_name})'

        self.min_value = np.nanmin(data_arr_all_incs)
        self.max_value = np.nanmax(data_arr_all_incs)
        self.min_value_inc = np.nanmin(data_arr)
        self.max_value_inc = np.nanmax(data_arr)
        self.min_value_inc_slice = np.nanmin(self.plot_data)
        self.max_value_inc_slice = np.nanmax(self.plot_data)
        self.xlabel = plane_data['xlabel']
        self.ylabel = plane_data['ylabel']
        self.title = title
        self.data_meta = self.get_field_metadata(data_name)

        # print(f'self.plot_data.shape: {self.plot_data.shape}')

    def _generate_figure(self):

        data = [
            {
                'type': 'heatmap',
                'z': self.plot_data[::-1],
                'zmin': self.min_value_inc_slice,
                'zmax': self.max_value_inc_slice,
                'colorscale': 'Viridis',
                'colorbar': {
                    'title': self.title,
                }
            },
            {
                # horizontal line profile
                'type': 'scatter',
                'x': [0, self.grid_size[0]],  # todo: fix
                'y': [self.line_hor_position] * 2,
                'mode': 'lines',
                'line': {
                    'width': 2,
                },
                'showlegend': False,
            },
            {
                # vertical line profile
                'type': 'scatter',
                'x': [self.line_ver_position] * 2,
                'y': [0, self.grid_size[0]],  # todo: fix
                'mode': 'lines',
                'line': {
                    'width': 2,
                },
                'showlegend': False,
            },
            {
                'x': np.arange(self.grid_size[0]),  # todo fix
                'y': np.random.randint(0, 9, (self.grid_size[0])),
                'xaxis': 'x2',
                'yaxis': 'y2',
            },
            {
                'x': np.random.randint(0, 9, (self.grid_size[0])),
                'y': np.arange(self.grid_size[0]),  # todo fix
                'xaxis': 'x3',
                'yaxis': 'y3',
            },
        ]
        layout = {
            'template': 'none',
            'paper_bgcolor': 'pink',
            'plot_bgcolor': 'green',
            'margin': {
                't': 50,
                'r': 0,
                'b': 50,
                'l': 0,
            },
            'height': 500,
            'width': 700,
            'showlegend': False,
            'xaxis': {
                'domain': [0.3, 0.8],
                'scaleanchor': 'y',
                'constrain': 'domain',
                'title': {'text': self.xlabel, 'font': {'size': 22}},
                'showgrid': False,
                'showticklabels': False,
                'ticks': '',
            },
            'yaxis': {
                'domain': [0.3, 0.8],
                'title': {'text': self.ylabel, 'font': {'size': 22}},
                'showgrid': False,
                'showticklabels': False,
                'ticks': '',
                # 'autorange': 'reversed', # todo: doesn't work when updating
            },
            'xaxis2': {
                'domain': [0.3, 0.8],
                # 'overlaying': 'y2',
                'scaleanchor': 'x',
                # 'matches': 'x',
            },
            'yaxis2': {
                'domain': [0.1, 0.25],
            },
            'xaxis3': {
                'domain': [0.1, 0.25],
                # 'overlaying': 'y3',
            },
            'yaxis3': {
                'domain': [0.3, 0.8],
                'scaleanchor': 'y',
                # 'matches': 'y',
            },
        }
        fig = graph_objects.FigureWidget(data=data, layout=layout)

        return fig

    def _generate_widgets(self):

        inc_control = ipywidgets.IntSlider(
            value=self.increment_idx,
            min=0,
            max=self.get_num_increments(self.data_name) - 1,
            step=1,
            description='Increment:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        data_control = ipywidgets.Dropdown(
            options=self.field_data_names,
            value=self.data_name,
            description='Data:',
        )
        data_component_control = ipywidgets.Dropdown(
            options=[str(i)
                     for i in self._get_allowed_tensorial_components(self.data_name)],
            value=str(self.data_component),
            description='Component:',
        )
        colourbar_limits_control = ipywidgets.RadioButtons(
            options=[
                'Visible range',
                'Current increment',
                'All increments',
            ],
            value='Visible range',
            description='Colour bar limits:',
        )
        callbacks_control = ipywidgets.Dropdown(
            options=['None'] + list(self.callbacks.keys()),
            value='None',
            description='Callback:',
        )
        line_hor_position_control = ipywidgets.IntSlider(
            value=0,
            min=0,
            max=self.grid_size[0],  # todo fix
            step=1,
            description='Line (hor.) pos.:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        line_ver_position_control = ipywidgets.IntSlider(
            value=0,
            min=0,
            max=self.grid_size[0],  # todo fix
            step=1,
            description='Line (ver.) pos.:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )

        inc_control.observe(self._update_increment, names='value')
        data_control.observe(self._update_data, names='value')
        data_component_control.observe(self._update_data_component, names='value')
        colourbar_limits_control.observe(self._update_colourbar_limits, names='value')
        callbacks_control.observe(self._update_callback, names='value')
        line_hor_position_control.observe(self._update_line_hor_position, names='value')
        line_ver_position_control.observe(self._update_line_ver_position, names='value')

        widgets = {
            'figure': self.figure,
            'increment_control': inc_control,
            'data_control': data_control,
            'data_component_control': data_component_control,
            'colourbar_limits_control': colourbar_limits_control,
            'callbacks_control': callbacks_control,
            'line_hor_position_control': line_hor_position_control,
            'line_ver_position_control': line_ver_position_control,

        }

        return widgets

    def _get_field_data_inner_shape(self, field_data_name):
        return self.get_field_data_at_increment(field_data_name, 0).shape[3:]

    def _get_allowed_tensorial_components(self, field_data_name):
        return list(np.ndindex(*self._get_field_data_inner_shape(field_data_name)))

    def get_field_data(self, name):
        if name == 'O':
            return self.field_data[name]['data']['quaternions']
        elif name == 'phase':
            return self.field_data[name]['data'][None]  # Add an increment dimension
        else:
            return self.field_data[name]['data']

    def get_field_metadata(self, name):
        return self.field_data[name]['meta']

    def get_field_data_at_increment(self, name, increment):
        return self.get_field_data(name)[increment]

    @property
    def field_data(self):
        """All field data excluding phase"""
        return self.volume_element_response['field_data']

    @property
    def field_data_names(self):
        return tuple(self.field_data.keys())

    @property
    def _first_field_data_name(self):
        return self.field_data_names[0]

    @property
    def _first_field_data(self):
        return self.field_data[self._first_field_data_name]

    @property
    def grid_size(self):
        return self._first_field_data['data'].shape[1:4]

    def get_num_increments(self, name):
        return len(self.get_increments(name))

    def get_increments(self, name):
        if name == 'phase':
            return [0]
        else:
            return self.field_data[name]['meta']['increments']

    def _update_increment(self, change):

        new_inc_idx = change['new']

        self.increment_idx = new_inc_idx

        self._set_plot_data(self.data_name, self.data_component,
                            new_inc_idx, self.current_callback)
        with self.figure.batch_update():
            self.figure.data[0].z = self.plot_data[::-1]
            self.figure.data[0].zmin, self.figure.data[0].zmax = self._get_colourbar_limits()

    def _update_data(self, change):

        new_data_name = change['new']

        allowed_comps = self._get_allowed_tensorial_components(new_data_name)
        if self.data_component not in allowed_comps:
            new_data_comp = allowed_comps[0]
        else:
            new_data_comp = self.data_component

        num_incs = self.get_num_increments(new_data_name)
        if self.increment_idx not in range(num_incs):
            new_inc_idx = 0
        else:
            new_inc_idx = self.increment_idx

        self._widgets['increment_control'].max = num_incs - 1
        if num_incs == 1:
            self._widgets['increment_control'].disabled = True
        else:
            self._widgets['increment_control'].disabled = False

        self.increment_idx = new_inc_idx
        self.data_name = new_data_name
        self.data_component = new_data_comp

        self._set_plot_data(new_data_name, new_data_comp,
                            new_inc_idx, self.current_callback)

        self._widgets['data_component_control'].options = [str(i) for i in allowed_comps]
        self._widgets['data_component_control'].value = str(new_data_comp)
        if len(allowed_comps) == 1:
            self._widgets['data_component_control'].disabled = True
        else:
            self._widgets['data_component_control'].disabled = False
        self._widgets['increment_control'].value = new_inc_idx

        with self.figure.batch_update():
            self.figure.data[0].z = self.plot_data[::-1]
            self.figure.data[0].zmin, self.figure.data[0].zmax = self._get_colourbar_limits()
            self.figure.data[0].colorbar.title = self.title

    def _update_data_component(self, change):

        new_comp_str = change['new']
        new_comp_csv = new_comp_str.split('(')[1].split(')')[0]
        if new_comp_csv:
            new_comp = tuple(int(i) for i in new_comp_csv.split(',') if i != '')
        else:
            new_comp = ()

        self.data_component = new_comp
        self._set_plot_data(self.data_name, new_comp,
                            self.increment_idx, self.current_callback)
        with self.figure.batch_update():
            self.figure.data[0].z = self.plot_data[::-1]
            self.figure.data[0].zmin, self.figure.data[0].zmax = self._get_colourbar_limits()
            self.figure.data[0].colorbar.title = self.title

    def _update_colourbar_limits(self, change):
        with self.figure.batch_update():
            self.figure.data[0].zmin, self.figure.data[0].zmax = self._get_colourbar_limits()

    def _update_callback(self, change):
        new_callback = change['new']
        try:
            self._set_plot_data(
                self.data_name,
                self.data_component,
                self.increment_idx,
                new_callback,
            )
        except ValueError:
            print('Reverting!')
            self._widgets['callbacks_control'].value = change['old']
            return
        self.current_callback = new_callback
        with self.figure.batch_update():
            self.figure.data[0].z = self.plot_data[::-1]
            self.figure.data[0].zmin, self.figure.data[0].zmax = self._get_colourbar_limits()
            self.figure.data[0].colorbar.title = self.title

    def _update_line_hor_position(self, change):
        new_hor_position = change['new']
        # second trace
        with self.figure.batch_update():
            self.figure.data[1].y = [new_hor_position] * 2

    def _update_line_ver_position(self, change):
        new_ver_position = change['new']
        # third trace
        with self.figure.batch_update():
            self.figure.data[2].x = [new_ver_position] * 2

    def _get_colourbar_limits(self):
        if self._widgets['colourbar_limits_control'].value == 'Visible range':
            zmin = self.min_value_inc_slice
            zmax = self.max_value_inc_slice
        elif self._widgets['colourbar_limits_control'].value == 'Current increment':
            zmin = self.min_value_inc
            zmax = self.max_value_inc
        elif self._widgets['colourbar_limits_control'].value == 'All increments':
            zmin = self.min_value
            zmax = self.max_value
        return zmin, zmax

    def visual(self):
        out = ipywidgets.HBox(children=[
            self._widgets['figure'],
            ipywidgets.VBox(children=[
                self._widgets['data_control'],
                self._widgets['data_component_control'],
                self._widgets['increment_control'],
                self._widgets['colourbar_limits_control'],
                self._widgets['callbacks_control'],
                self._widgets['line_hor_position_control'],
                self._widgets['line_ver_position_control'],
            ])
        ])
        return out

    def show(self):
        return self.visual()


def visualise_field_data(volume_element_response, increment=None, x=None, y=None, z=None,
                         transpose=False, data_name=None, data_component=None,
                         global_colour_scale=True, callbacks=None):

    RVE_field_viz = RVEFieldViz(
        volume_element_response,
        increment=increment,
        x=x,
        y=y,
        z=z,
        transpose=transpose,
        data_name=data_name,
        data_component=data_component,
        global_colour_scale=global_colour_scale,
        callbacks=callbacks,
    )
    return RVE_field_viz
