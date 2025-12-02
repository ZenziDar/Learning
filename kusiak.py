import streamlit as st
import pandas as pd
import numpy as np

def kusiak_method(incidence_matrix):
    """
    Applies the Kusiak Method (Linear Clustering Algorithm) to an incidence matrix
    to identify manufacturing cells and part families.

    Parameters:
    - incidence_matrix (pd.DataFrame): A DataFrame where index=Machines, columns=Parts.
                                        Contains 0s and 1s.

    Returns:
    - list: A list of dictionaries, where each dict represents an identified cell.
    """
    # Use a copy to avoid modifying the original data
    remaining_matrix = incidence_matrix.copy()
    cells = []
    cell_id = 1

    # Step 6: Repeat until the matrix is empty
    while not remaining_matrix.empty and np.any(remaining_matrix.values == 1):
        # Step 1: Select any row (Machine) with at least one '1'
        machines_with_one = remaining_matrix.index[remaining_matrix.sum(axis=1) > 0]
        if machines_with_one.empty:
            break  # No more '1's left

        selected_machine = machines_with_one[0]
        current_machines = {selected_machine}
        current_parts = set()
        
        # Tracking set sizes to check for stabilization (required for any dimension)
        previous_machines_size = 0
        previous_parts_size = 0

        # Iterative line drawing simulation (Steps 2, 3, 4)
        # Continue as long as the size of either set is increasing.
        while len(current_machines) != previous_machines_size or len(current_parts) != previous_parts_size:
            
            # Update previous sizes for the next check
            previous_machines_size = len(current_machines)
            previous_parts_size = len(current_parts)
            
            # Step 2: For current machines, find all connected parts
            newly_added_parts = set()
            valid_machines = [m for m in current_machines if m in remaining_matrix.index]
            
            if valid_machines:
                # Sum columns for the selected rows. Parts with sum > 0 are connected.
                parts_series = remaining_matrix.loc[valid_machines].sum(axis=0)
                newly_added_parts.update(parts_series[parts_series > 0].index.tolist())
            
            current_parts.update(newly_added_parts)

            # Step 3: For current parts, find all connected machines
            newly_added_machines = set()
            valid_parts = [p for p in current_parts if p in remaining_matrix.columns]
            
            if valid_parts:
                # Sum rows for the selected columns. Machines with sum > 0 are connected.
                machines_series = remaining_matrix[valid_parts].sum(axis=1)
                newly_added_machines.update(machines_series[machines_series > 0].index.tolist())

            current_machines.update(newly_added_machines)
            
        # Step 5: Form a cell
        cell = {
            'id': cell_id,
            'machines': sorted(list(current_machines)),
            'parts': sorted(list(current_parts))
        }
        cells.append(cell)

        # Step 6: Remove the used elements
        machines_to_drop = [m for m in current_machines if m in remaining_matrix.index]
        parts_to_drop = [p for p in current_parts if p in remaining_matrix.columns]

        # Use errors='ignore' for robust dropping
        remaining_matrix = remaining_matrix.drop(index=machines_to_drop, errors='ignore')
        remaining_matrix = remaining_matrix.drop(columns=parts_to_drop, errors='ignore')
        
        cell_id += 1
        
    return cells

def create_rearranged_styler(df, cells):
    """
    Creates a styled rearranged block diagonal matrix.
    - Groups machines and parts by cell (cluster).
    - Applies very light background to entire cluster blocks for visibility.
    - Bolds and darkens text for 1s to ensure clarity.
    - Colors inter-cluster 1s (exceptions) in light red.
    """
    if not cells:
        return df.style
    
    machine_to_cell = {}
    part_to_cell = {}
    for cell in cells:
        for m in cell['machines']:
            machine_to_cell[m] = cell['id']
        for p in cell['parts']:
            part_to_cell[p] = cell['id']
    
    all_machines = list(df.index)
    all_parts = list(df.columns)
    sorted_machines = sorted(all_machines, key=lambda x: machine_to_cell.get(x, 0))
    sorted_parts = sorted(all_parts, key=lambda x: part_to_cell.get(x, 0))
    rearranged = df.loc[sorted_machines, sorted_parts].copy()
    
    # Clearer, very light colors for blocks
    colors = [
        "#0568E9",  # Very light blue
        "#A90DE6",  # Very light purple
        "#15EB15",  # Very light green
        "#EE6C0A",  # Very light orange
        "#EF0FAF",  # Very light pink
        "#3A8A46",  # Very light mint
        "#F6FF00",  # Very light yellow
        "#F0105B"   # Very light lime
    ]
    
    def highlight(x):
        styles = pd.DataFrame('', index=x.index, columns=x.columns)
        for i in x.index:
            for j in x.columns:
                val = x.loc[i, j]
                m_cell = machine_to_cell.get(i, 0)
                p_cell = part_to_cell.get(j, 0)
                
                if m_cell == p_cell and m_cell > 0:
                    # Light block background for entire cluster
                    color = colors[(m_cell - 1) % len(colors)]
                    style = f'background-color: {color}'
                    
                    # If val == 1, bold and darken text for clarity
                    if val == 1:
                        style += '; font-weight: bold; color: #000000'
                    
                    styles.loc[i, j] = style
                elif val == 1:
                    # Exceptions: light red background, bold
                    styles.loc[i, j] = 'background-color: #FFF2F2; font-weight: bold; color: #000000'
        
        return styles
    
    styler = rearranged.style.apply(highlight, axis=None).set_properties(**{
        'text-align': 'center',
        'font-size': '12px'
    })
    return styler

# Streamlit App
st.title("Kusiak Method for Manufacturing Cell Formation")
st.markdown("Upload or edit an incidence matrix (rows: Machines, columns: Parts, values: 0/1) to identify cells and part families.")

# Sample data
@st.cache_data
def load_sample_data():
    product_columns = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9']
    machine_data_rows = {
        'M1':  [0, 0, 0, 0, 1, 0, 1, 0, 0],
        'M2':  [0, 0, 0, 0, 0, 1, 0, 0, 1],
        'M3':  [0, 1, 1, 0, 0, 0, 1, 0, 0],
        'M4':  [0, 0, 0, 1, 0, 0, 0, 1, 0],
        'M5':  [0, 1, 1, 0, 1, 0, 0, 0, 0],
        'M6':  [0, 0, 0, 1, 0, 1, 0, 0, 1],
        'M7':  [1, 0, 0, 0, 0, 0, 0, 1, 0],
        'M8':  [0, 0, 0, 1, 0, 1, 0, 1, 1],
        'M9':  [0, 1, 1, 0, 1, 0, 1, 0, 0],
        'M10': [1, 0, 0, 0, 0, 0, 0, 1, 0]
    }
    return pd.DataFrame.from_dict(machine_data_rows, orient='index', columns=product_columns)

# Load sample or upload
uploaded_file = st.file_uploader("Upload CSV (optional; use sample below if none)", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, index_col=0)
    st.success("File uploaded successfully!")
else:
    df = load_sample_data()
    st.info("Using sample data.")

# Display and edit matrix
st.subheader("Incidence Matrix")
st.write("Edit the matrix below (0/1 values only).")
edited_df = st.data_editor(
    df,
    column_config={
        col: st.column_config.NumberColumn(
            col,
            help="0 or 1",
            min_value=0,
            max_value=1,
            step=1,
            format="%d"
        ) for col in df.columns
    },
    disabled=False,
    hide_index=False,
    use_container_width=True
)

# Run button
if st.button("Run Kusiak Method", type="primary"):
    if edited_df.isnull().values.any():
        st.error("Matrix contains NaN values. Please fill with 0 or 1.")
    else:
        with st.spinner("Running Kusiak Method..."):
            results = kusiak_method(edited_df)
        
        if results:
            st.success(f"Identified {len(results)} cell(s).")
            for cell in results:
                with st.expander(f"Cell {cell['id']}: {len(cell['machines'])} Machines, {len(cell['parts'])} Parts"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Machines")
                        st.write(cell['machines'])
                    with col2:
                        st.subheader("Parts")
                        st.write(cell['parts'])
                    
                    # Sub-matrix
                    sub_df = edited_df.loc[cell['machines'], cell['parts']].fillna(0)
                    st.subheader("Sub-Matrix")
                    st.dataframe(sub_df, use_container_width=True)
            
            # Rearranged Block Diagonal Matrix
            st.subheader("Rearranged Block Diagonal Matrix (Colored Clusters)")
            st.markdown("""
            **Legend:** 
            - Each cluster (cell) block has a very light unique background for easy identification. 
            - 1s within clusters are bolded and black for maximum clarity. 
            - Inter-cluster 1s (exceptions) are light red and bolded. 
            - 0s remain fully visible with no obstruction.
            """)
            styler = create_rearranged_styler(edited_df, results)
            st.dataframe(styler, use_container_width=True)
        else:
            st.warning("No cells identified. Check if matrix has any 1s.")

# Footer
st.markdown("---")
st.markdown("**Notes:** This app uses the standard Kusiak linear clustering algorithm. Cells are non-overlapping connected components.")