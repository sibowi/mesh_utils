import plotly
import plotly.graph_objects as go
import numpy as np

def create_figure_for_inspection(mesh, path_figure):

    plot_mesh = get_mesh_plot([mesh.vertices], [mesh.faces], show_edges=True)
    
    #### Set layout
    layout = go.Layout(
        title=f'',
        font=dict(size=16, color='white'),
        width=1200,
        height=1000,
        paper_bgcolor='rgb(50,50,50)',
        scene_aspectmode='data',
        scene=dict(#aspectmode='data',
            xaxis = dict(
                title='x [mm]',
                backgroundcolor='rgb(50,50,50)',
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",),
            yaxis = dict(
                title='y [mm]',
                backgroundcolor='rgb(50,50,50)',
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white"),
            zaxis = dict(
                title='z [mm]',
                backgroundcolor='rgb(50,50,50)',
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white")
        ),
    )

    #### Create figure
    fig = go.Figure(data=plot_mesh, layout=layout)

    #### Show and save figure
    fig.write_html(path_figure, include_mathjax='cdn')

    print(f'Figure was saved to {path_figure}. Check it out in your browser (Firefox might work better).')

def get_mesh_plot(vertices_all, faces_all, face_color='rgb(240, 240, 240)', show_edges=True):
    """
    TBW
    """
    assert len(np.shape(vertices_all)) == len(np.shape(vertices_all))

    if (len(np.shape(vertices_all)) == 2):
        vertices_all = [vertices_all]
        faces_all = [faces_all]
    elif (len(np.shape(vertices_all)) == 1):
        pass
    elif (len(np.shape(vertices_all)) == 3):
        # TODO: What's going on here
        pass
    else:
        print(f'Something with shape of inputs...')
        exit()

    data = []

    for vertices, faces in zip(vertices_all, faces_all):

        x, y, z = vertices.T
        I, J, K = faces.T

        tri_points = vertices[faces]

        #pl_mygrey=[0, 'rgb(153, 153, 153)'], [1., 'rgb(153, 153, 153)']
        pl_mygrey=[0, face_color], [1., face_color]

        pl_mesh = go.Mesh3d(x=x,
                            y=y,
                            z=z,
                            colorscale=pl_mygrey,
                            intensity= z,
                            flatshading=True,
                            i=I,
                            j=J,
                            k=K,
                            name='Mesh',
                            showscale=False,
                            opacity=0.8,
                            )

        #### Extract data to plot the triangle edges as lines
        Xe = []
        Ye = []
        Ze = []
        for T in tri_points:
            Xe.extend([T[k%3][0] for k in range(4)]+[ None])
            Ye.extend([T[k%3][1] for k in range(4)]+[ None])
            Ze.extend([T[k%3][2] for k in range(4)]+[ None])

        #### Define the trace for triangle sides
        lines = go.Scatter3d(
                           x=Xe,
                           y=Ye,
                           z=Ze,
                           mode='lines',
                           name='',
                           line=dict(color= 'rgb(70,70,70)', width=1))

        data.append(pl_mesh)

        if show_edges == True:
            data.append(lines)

    return data


def plot_mesh(vertices_all, faces_all, path_figures=None, figure_name=None, show_edges=True):
    """
    TBW
    """
    assert len(np.shape(vertices_all)) == len(np.shape(vertices_all))

    if (len(np.shape(vertices_all)) == 2):
        vertices_all = [vertices_all]
        faces_all = [faces_all]
    elif (len(np.shape(vertices_all)) == 1):
        pass
    elif (len(np.shape(vertices_all)) == 3):
        # TODO: What's going on here
        pass
    else:
        print(f'Something with shape of inputs...')
        exit()

    data = []

    for vertices, faces in zip(vertices_all, faces_all):

        x, y, z = vertices.T
        I, J, K = faces.T

        tri_points = vertices[faces]

        pl_mygrey=[0, 'rgb(153, 153, 153)'], [1., 'rgb(255,255,255)']

        pl_mesh = go.Mesh3d(x=x,
                            y=y,
                            z=z,
                            colorscale=pl_mygrey,
                            intensity= z,
                            flatshading=True,
                            i=I,
                            j=J,
                            k=K,
                            name='Mesh',
                            showscale=False,
                            opacity=0.5,
                            )

        #### Extract data to plot the triangle edges as lines
        Xe = []
        Ye = []
        Ze = []
        for T in tri_points:
            Xe.extend([T[k%3][0] for k in range(4)]+[ None])
            Ye.extend([T[k%3][1] for k in range(4)]+[ None])
            Ze.extend([T[k%3][2] for k in range(4)]+[ None])

        #### Define the trace for triangle sides
        lines = go.Scatter3d(
                           x=Xe,
                           y=Ye,
                           z=Ze,
                           mode='lines',
                           name='',
                           line=dict(color= 'rgb(70,70,70)', width=1))

        #### Set layout
        layout = go.Layout(
             title="",
             font=dict(size=16, color='white'),
             width=1200,
             height=1200,
             #scene_xaxis_visible=False,
             #scene_yaxis_visible=False,
             #scene_zaxis_visible=False,
             paper_bgcolor='rgb(50,50,50)',

            )

        data.append(pl_mesh)

        if show_edges == True:
            data.append(lines)


    fig = go.Figure(data=data, layout=layout)

    if (path_figures != None) and (figure_name != None):
        plotly.offline.plot(fig, filename=path_figures+figure_name, include_mathjax='cdn')
    else:
        fig.show()
