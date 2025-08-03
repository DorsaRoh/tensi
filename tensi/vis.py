import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.colors as colors
from typing import Any

class TensorVisualizer:
    def __init__(self, colorscale = 'Blues'):
        self.colorscale = colorscale  # Can be changed to RdBu, Blues, etc.
    
    def visualize_tensor(self, tensor: Any, dtype: str, title: str):
        """
        visualization based on tensor dimensions:
        - 3D [B, S, D]: 2D grids (S×D) for each batch
        - 4D [B, W, H, D]: 3D cubes (W×H×D) for each batch
        """
        dtype = getattr(torch, dtype)  # convert dtype(string) to torch.dtype
        tensor = torch.tensor(tensor, dtype=dtype)  # convert user tensor to torch tensor
        original_shape = tensor.shape
        
        if len(original_shape) == 2:    # originally 2d tensor. convert to 3d (add batch dimension)
            tensor = tensor.unsqueeze(0)    # ex. [[1,2],[3,4]] -> [[[1,2],[3,4]]] | shape: [X, Y] -> [B, X, Y]
            #print("Unsqueezed: ", tensor.shape)
            return self._visualize_3d_as_2d_grids(tensor, title, original_shape)
        elif len(original_shape) == 3:
            # [B, S, D] -> 2D grids
            return self._visualize_3d_as_2d_grids(tensor, title, original_shape)
        elif len(original_shape) == 4:
            # [B, W, H, D] -> 3D cubes
            return self._visualize_4d_as_3d_cubes(tensor, title, original_shape)
        # else:
        #     # Handle other dimensions by reshaping
        #     return self._visualize_fallback(tensor, title, original_shape)
    
    def _visualize_3d_as_2d_grids(self, tensor, title, original_shape):
        """
        Visualize 3D tensor [B, S, D] as 2D grids
        Each batch gets a 2D heatmap grid where S×D forms the plane
        """
        print("Original shape", original_shape)
        print(tensor.shape)

        if len(original_shape) == 2:    # originally 2D tensor
            B, S, D = 1, tensor.shape[1], tensor.shape[2]
        else:  
            B, S, D = tensor.shape
        
        # create subplot grid layout for displaying BATCHES
        subplot_cols = min(B, 4)    # max 4 columns in the subplot grid / i.e. max 4 batches shown
        subplot_rows = (B + subplot_cols - 1) // subplot_cols  # ceiling division

        print("subplot_cols: ", subplot_cols)
        print("subplot_rows: ", subplot_rows)
        
        fig = make_subplots(
            rows=subplot_rows, cols=subplot_cols,
            subplot_titles=[f'Batch {i+1}' for i in range(B)],
            specs=[[{'type': 'heatmap'} for _ in range(subplot_cols)] for _ in range(subplot_rows)],
            horizontal_spacing=0.2,
            vertical_spacing=0.2
        )
        
        for b in range(B):  # for each batch
            # calculate position of current cell value (x, y) (plotly uses 1-indexed)
            pos_x = (b // subplot_cols) + 1    # calculate which row in batch grid
            pos_y = (b % subplot_cols) + 1     # calculate which column in batch grid
            
            batch_data = (tensor[0]).numpy()  # convert batch tensor to numpy tensor
            print(batch_data)
            
            # create heatmap
            fig.add_trace(
                go.Heatmap(
                    z=batch_data,
                    colorscale=self.colorscale,
                    showscale=(True),  # Only show colorbar on last subplot
                    hovertemplate=f'Batch {b+1}<br>S: %{{y}}<br>D: %{{x}}<br>Value: %{{z:.3f}}<extra></extra>',
                    name=f'Batch {b+1}'
                ),
                row=pos_x, col=pos_y
            )
            
            # update axes labels
            fig.update_xaxes(title_text="dimension 2", row=pos_x, col=pos_y)
            fig.update_yaxes(title_text="dimension 1", row=pos_x, col=pos_y)
        
        # Dynamic layout based on tensor dimensions and subplot structure
        base_cell_size = 40  # Base size per data cell
        margin = 120  # Space for titles, labels, colorbar
        
        # Calculate dynamic dimensions
        dynamic_height = max(300, S * base_cell_size + margin) * subplot_rows
        dynamic_width = max(250, D * base_cell_size + margin) * subplot_cols
        
        fig.update_layout(
            title=f"{title}<br>Shape: {list(original_shape)} | Batches: {B}",
            height=dynamic_height,
            width=dynamic_width,
            showlegend=False  # Hide legend for cleaner look
        )
        
        return fig
    
    # pick a margin that scales with cube size
    def _annotate_cube(self, fig, x0, y0, z0, W, H, D, fontsize=14):
        pad = 0.15 * max(W, H, D)        # Δ in scene units – large enough to escape the mesh

        # x ticks (0‥W) – run along the front-bottom edge
        for w in range(W + 1):
            fig.add_trace(go.Scatter3d(
                x=[x0 + w],
                y=[y0 - pad],             # pull forward in +/-y
                z=[z0 - pad],             # …and down a touch in z so text floats in front
                mode="text",
                text=[str(w)],
                textfont=dict(size=fontsize, color="black", family="Courier New"),
                hoverinfo="skip", showlegend=False))

        # y ticks (0‥H) – up the left front edge
        for h in range(H + 1):
            fig.add_trace(go.Scatter3d(
                x=[x0 - pad],
                y=[y0 + h],
                z=[z0 - pad],
                mode="text",
                text=[str(h)],
                textfont=dict(size=fontsize, color="black", family="Courier New"),
                hoverinfo="skip", showlegend=False))

        # z ticks (0‥D) – depth axis pointing away from viewer
        for d in range(D + 1):
            fig.add_trace(go.Scatter3d(
                x=[x0 - pad],
                y=[y0 - pad],
                z=[z0 + d],
                mode="text",
                text=[str(d)],
                textfont=dict(size=fontsize, color="black", family="Courier New"),
                hoverinfo="skip", showlegend=False))

    def _visualize_4d_as_3d_cubes(self, tensor, title, _):
        B, W, H, D = tensor.shape
        fig = go.Figure()

        for b in range(B):
            x_off = b * (W + 2)
            self.add_batch_cube(fig, x_off, 0, 0, W, H, D, f"batch {b+1}")
            self._add_internal_grid(fig,   x_off, 0, 0, W, H, D)
            self._annotate_cube(fig,       x_off, 0, 0, W, H, D)  # <-- call *after* cube/grid traces

        # hide global scene grid
        axis_nogrid = dict(showgrid=False, zeroline=False, showticklabels=False)
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=axis_nogrid, yaxis=axis_nogrid, zaxis=axis_nogrid,
                camera=dict(eye=dict(x=2, y=1.8, z=1.6)),
                aspectmode='manual', aspectratio=dict(x=2, y=1, z=1)),
            width=900, height=600)

        return fig
    
    def visualize_batch(self, fig, x, y, z, value, normalized_value, hover_text):
        """Visualize each batch"""
        # Define cube vertices
        vertices = np.array([
            [x, y, z], [x+0.8, y, z], [x+0.8, y+0.8, z], [x, y+0.8, z],           # bottom
            [x, y, z+0.8], [x+0.8, y, z+0.8], [x+0.8, y+0.8, z+0.8], [x, y+0.8, z+0.8]  # top
        ])
        
        # Get color
        color = self._get_color(normalized_value)
        
        # Create mesh3d cube
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=[0, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[1, 2, 3, 4, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[2, 3, 4, 5, 6, 7, 1, 1, 5, 5, 7, 6],
            color=color,
            opacity=0.8,
            showscale=False,
            hovertemplate=hover_text + "<extra></extra>",
            showlegend=False
        ))
    
    # ------------------------------------------------ patch
    def add_batch_cube(self, fig, x0, y0, z0, W, H, D, batch_name):
        # 8 vertices, numbered as in the diagram below
        #      7──────6        z ↑
        #     ╱|     ╱|        y ↗
        #    4──────5 |        x →
        #    | 3────| 2
        #    |╱     |╱
        #    0──────1
        xs = [x0, x0+W, x0+W, x0,  x0, x0+W, x0+W, x0]
        ys = [y0, y0,   y0+H, y0+H, y0, y0,   y0+H, y0+H]
        zs = [z0, z0,   z0,   z0,   z0+D, z0+D, z0+D, z0+D]

        # quads for the 6 faces (v0-v1-v2-v3)
        quads = [
            (0,1,2,3),  # bottom
            (4,5,6,7),  # top
            (0,1,5,4),  # front
            (1,2,6,5),  # right
            (2,3,7,6),  # back
            (3,0,4,7)   # left
        ]

        # build triangles; add both orientations → no visible holes
        tri_i, tri_j, tri_k = [], [], []
        for a,b,c,d in quads:
            # outward
            tri_i += [a, a]
            tri_j += [b, c]
            tri_k += [c, d]
            # inward (prevents back-face culling gaps)
            tri_i += [a, a]
            tri_j += [c, b]
            tri_k += [d, c]

        fig.add_trace(go.Mesh3d(
            x=xs, y=ys, z=zs,
            i=tri_i, j=tri_j, k=tri_k,
            color='lightblue',
            opacity=1.0,
            flatshading=True,
            lighting=dict(ambient=1, diffuse=0, specular=0, roughness=1),
            hoverinfo='skip',
            showscale=False,
            showlegend=False
        ))

        # thick black outer edges (unchanged)
        edges = [(0,1),(1,2),(2,3),(3,0),
                (4,5),(5,6),(6,7),(7,4),
                (0,4),(1,5),(2,6),(3,7)]
        for u,v in edges:
            fig.add_trace(go.Scatter3d(
                x=[xs[u],xs[v]], y=[ys[u],ys[v]], z=[zs[u],zs[v]],
                mode='lines', line=dict(color='black', width=3),
                hoverinfo='skip', showlegend=False))
    # ------------------------------------------------ patch end
        
        
    def _add_internal_grid(self, fig, x0, y0, z0, W, H, D):
        """draws grid lines on each cube face so the subdivision is visible even with opaque faces"""
        # helper to drop a line once
        def _line(x, y, z):
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color='darkgray', width=2),
                hoverinfo='skip',
                showlegend=False
            ))

        # front  (z = z0) & back (z = z0 + D) faces
        for z in [z0, z0 + D]:
            for w in range(1, W):
                _line([x0 + w, x0 + w], [y0, y0 + H], [z, z])
            for h in range(1, H):
                _line([x0, x0 + W], [y0 + h, y0 + h], [z, z])

        # left   (y = y0) & right (y = y0 + H) faces
        for y in [y0, y0 + H]:
            for w in range(1, W):
                _line([x0 + w, x0 + w], [y, y], [z0, z0 + D])
            for d in range(1, D):
                _line([x0, x0 + W], [y, y], [z0 + d, z0 + d])

        # bottom (x = x0) & top  (x = x0 + W) faces
        for x in [x0, x0 + W]:
            for h in range(1, H):
                _line([x, x], [y0 + h, y0 + h], [z0, z0 + D])
            for d in range(1, D):
                _line([x, x], [y0, y0 + H], [z0 + d, z0 + d])

    def _get_color(self, normalized_value):
        """Convert normalized value (0-1) to color"""
        colorscale = colors.sample_colorscale(self.colorscale, [normalized_value])[0]
        return colorscale
    
    # def _visualize_fallback(self, tensor, title, original_shape):
    #     """Fallback for other tensor dimensions"""
    #     # Reshape to 4D and use 3D cube visualization
    #     if len(original_shape) == 1:
    #         # [N] -> [1, 1, 1, N]
    #         reshaped = tensor.view(1, 1, 1, -1)
    #     elif len(original_shape) == 5:
    #         # [B, C, W, H, D] -> [B, C*W, H, D]
    #         B = original_shape[0]
    #         reshaped = tensor.view(B, -1, original_shape[-2], original_shape[-1])
    #     else:
    #         # General case: flatten middle dimensions
    #         B = original_shape[0] if len(original_shape) > 1 else 1
    #         reshaped = tensor.view(B, -1, 1, 1)
        
    #     return self._visualize_4d_as_3d_cubes(reshaped, f"{title} (Reshaped)", original_shape)

# ============ DEMO ============

def demo():
    tensorVisualizer = TensorVisualizer()
    
    # print("=== Demo 1: 2D tensor [3,2] → 2D Grid ===")
    # tensor_2d = [[1.2, 2.3], [3.1, 4.3], [5.2, 6.2]]
    dtype = "float32"
    # #print("Shape:", tensor_2d.shape)
    # fig_2d = tensorVisualizer.visualize_tensor(tensor_2d, dtype, "My tensor")
    # fig_2d.show()
    
    # print("\n=== Demo 2: 3D tensor [2,3,4] → 2D Grids ===")
    # tensor_3d = torch.randn(2, 3, 4)
    # dtype= "float32"
    # print("Shape:", tensor_3d.shape)
    # fig_3d = tensorVisualizer.visualize_tensor(tensor_3d, dtype, "my tensor")
    # fig_3d.show()
    
    print("\n=== Demo 3: 4D tensor [2,3,4,5] → 3D Cubes ===")
    tensor_4d = torch.tensor([
        [  # batch 0
            [  # width 0
                [  1,  2,  3,  4],     # height 0
                [  5,  6,  7,  8],     # height 1
                [  9, 10, 11, 12]      # height 2
            ],
            [  # width 1
                [ 13, 14, 15, 16],
                [ 17, 18, 19, 20],
                [ 21, 22, 23, 24]
            ]
        ],
        [  # batch 1
            [  # width 0
                [101, 102, 103, 104],
                [105, 106, 107, 108],
                [109, 110, 111, 112]
            ],
            [  # width 1
                [113, 114, 115, 116],
                [117, 118, 119, 120],
                [121, 122, 123, 124]
            ]
        ],
        [  # batch 1
            [  # width 0
                [101, 102, 103, 104],
                [105, 106, 107, 108],
                [109, 110, 111, 112]
            ],
            [  # width 1
                [113, 114, 115, 116],
                [117, 118, 119, 120],
                [121, 122, 123, 124]
            ]
        ]
    ], dtype=torch.float32)
    print("Shape:", tensor_4d.shape)
    fig_4d = tensorVisualizer.visualize_tensor(tensor_4d, dtype, "4D Tensor [B,W,H,D]")
    fig_4d.show()
    
    # print("\n=== Demo 4: Your original example [2,2] ===")
    # user_tensor = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float32)
    # print("Shape:", user_tensor.shape)
    # fig_user = viz.visualize_tensor(user_tensor, "Your Original Tensor")
    # fig_user.show()

    # 5D and above tensors

if __name__ == "__main__":
    demo()