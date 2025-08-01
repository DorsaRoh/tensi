import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.colors as colors
from typing import Any

class TensorVisualizer:
    def __init__(self, colorscale = 'Viridis'):
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
        # elif len(original_shape) == 4:
        #     # [B, W, H, D] -> 3D cubes
        #     return self._visualize_4d_as_3d_cubes(tensor, title, original_shape)
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
        
        # Create subplot grid layout for displaying BATCHES
        subplot_cols = min(B, 4)    # max 4 columns in the subplot grid / i.e. max 4 batches shown
        subplot_rows = (B + subplot_cols - 1) // subplot_cols  # ceiling division

        print("subplot_cols: ", subplot_cols)
        print("subplot_rows: ", subplot_rows)
        
        fig = make_subplots(
            rows=subplot_rows, cols=subplot_cols,
            subplot_titles=[f'Batch {i+1}' for i in range(B)],
            specs=[[{'type': 'heatmap'} for _ in range(subplot_cols)] for _ in range(subplot_rows)],
            horizontal_spacing=0.1,
            vertical_spacing=0.1
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
            fig.update_xaxes(title_text="D (Dimension)", row=pos_x, col=pos_y)
            fig.update_yaxes(title_text="S (Sequence)", row=pos_x, col=pos_y)
        
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
    
    # def _visualize_4d_as_3d_cubes(self, tensor, title, original_shape):
    #     """
    #     Visualize 4D tensor [B, W, H, D] as 3D cubes
    #     Each batch gets a 3D cube where W×H×D forms the volume
    #     """
    #     B, W, H, D = tensor.shape
        
    #     fig = go.Figure()
        
    #     # Get global min/max for consistent coloring
    #     global_min = tensor.min().item()
    #     global_max = tensor.max().item()
        
    #     # Create 3D cubes for each batch
    #     for batch_idx in range(B):
    #         batch_data = tensor[batch_idx]  # Shape: [W, H, D]
            
    #         # Offset each batch in space
    #         x_batch_offset = batch_idx * (W + 3)  # Space between batches
            
    #         # Create individual cubes for each tensor element
    #         for w in range(W):
    #             for h in range(H):
    #                 for d in range(D):
    #                     value = batch_data[w, h, d].item()
                        
    #                     # Position of this cube
    #                     x_pos = x_batch_offset + w
    #                     y_pos = h
    #                     z_pos = d
                        
    #                     # Normalize value for color mapping
    #                     if global_max != global_min:
    #                         normalized_value = (value - global_min) / (global_max - global_min)
    #                     else:
    #                         normalized_value = 0.5
                        
    #                     # Create cube
    #                     self._add_3d_cube(
    #                         fig, x_pos, y_pos, z_pos,
    #                         value, normalized_value,
    #                         f"Batch {batch_idx+1}<br>W: {w}<br>H: {h}<br>D: {d}<br>Value: {value:.3f}"
    #                     )
        
    #     # Update layout for 3D viewing
    #     fig.update_layout(
    #         title=f"{title}<br>Original Shape: {list(original_shape)} → 3D Cubes [W×H×D]",
    #         scene=dict(
    #             xaxis_title="W (Width) + Batch Offset",
    #             yaxis_title="H (Height)",
    #             zaxis_title="D (Depth)",
    #             camera=dict(
    #                 eye=dict(x=1.8, y=1.8, z=1.5)
    #             ),
    #             aspectmode='manual',
    #             aspectratio=dict(x=2, y=1, z=1)
    #         ),
    #         width=900,
    #         height=700
    #     )
        
    #     return fig
    
    # def _add_3d_cube(self, fig, x, y, z, value, normalized_value, hover_text):
    #     """Add a single colored cube to the 3D figure"""
    #     # Define cube vertices
    #     vertices = np.array([
    #         [x, y, z], [x+0.8, y, z], [x+0.8, y+0.8, z], [x, y+0.8, z],           # bottom
    #         [x, y, z+0.8], [x+0.8, y, z+0.8], [x+0.8, y+0.8, z+0.8], [x, y+0.8, z+0.8]  # top
    #     ])
        
    #     # Get color
    #     color = self._get_color(normalized_value)
        
    #     # Create mesh3d cube
    #     fig.add_trace(go.Mesh3d(
    #         x=vertices[:, 0],
    #         y=vertices[:, 1],
    #         z=vertices[:, 2],
    #         i=[0, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
    #         j=[1, 2, 3, 4, 5, 6, 5, 2, 0, 1, 6, 3],
    #         k=[2, 3, 4, 5, 6, 7, 1, 1, 5, 5, 7, 6],
    #         color=color,
    #         opacity=0.8,
    #         showscale=False,
    #         hovertemplate=hover_text + "<extra></extra>",
    #         showlegend=False
    #     ))
    
    # def _get_color(self, normalized_value):
    #     """Convert normalized value (0-1) to color"""
    #     colorscale = colors.sample_colorscale(self.colorscale, [normalized_value])[0]
    #     return colorscale
    
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
    
    print("=== Demo 1: 2D tensor [3,2] → 2D Grid ===")
    tensor_2d = [[1.2, 2.3], [3.1, 4.3], [5.2, 6.2]]
    dtype = "float32"
    #print("Shape:", tensor_2d.shape)
    fig_2d = tensorVisualizer.visualize_tensor(tensor_2d, dtype, "My tensor")
    fig_2d.show()
    
    print("\n=== Demo 2: 3D tensor [2,3,4] → 2D Grids ===")
    tensor_3d = torch.randn(2, 3, 4)
    dtype= "float32"
    print("Shape:", tensor_3d.shape)
    fig_3d = tensorVisualizer.visualize_tensor(tensor_3d, dtype, "my tensor")
    fig_3d.show()
    
    # print("\n=== Demo 3: 4D tensor [2,3,3,2] → 3D Cubes ===")
    # tensor_4d = torch.randn(2, 3, 3, 2)
    # print("Shape:", tensor_4d.shape)
    # fig_4d = viz.visualize_tensor(tensor_4d, "4D Tensor [B,W,H,D]")
    # fig_4d.show()
    
    # print("\n=== Demo 4: Your original example [2,2] ===")
    # user_tensor = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float32)
    # print("Shape:", user_tensor.shape)
    # fig_user = viz.visualize_tensor(user_tensor, "Your Original Tensor")
    # fig_user.show()

if __name__ == "__main__":
    demo()