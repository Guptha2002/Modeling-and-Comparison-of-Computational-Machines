import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import random
import time
from datetime import datetime
import pandas as pd
from io import BytesIO
import base64

# Page configuration
st.set_page_config(
    page_title="Computational Automata Simulator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

class DFA:
    def __init__(self):
        self.states = {'q0', 'q1'}
        self.alphabet = {'0', '1'}
        self.start_state = 'q0'
        self.accept_states = {'q0'}
        self.transitions = {
            'q0': {'0': 'q1', '1': 'q0'},
            'q1': {'0': 'q0', '1': 'q1'}
        }
        self.positions = {
            'q0': (150, 150),
            'q1': (350, 150)
        }

    def get_description(self):
        return "DFA: L = {w | w contains an even number of zeros}"

    def generate_random(self):
        length = random.randint(1, 6)
        return ''.join(random.choice(['0', '1']) for _ in range(length))

    def process(self, input_string):
        current_state = self.start_state
        steps = []
        state_sequence = [current_state]
        
        steps.append({'type': 'initial', 'content': f"Initial state: {current_state}"})

        for i, symbol in enumerate(input_string):
            if symbol not in self.alphabet:
                steps.append({'type': 'error', 'content': f"Error: Invalid symbol '{symbol}'"})
                return False, steps, state_sequence

            next_state = self.transitions[current_state][symbol]
            steps.append({
                'type': 'step',
                'number': i + 1,
                'current_state': current_state,
                'symbol': symbol,
                'next_state': next_state
            })
            current_state = next_state
            state_sequence.append(current_state)

        accepted = current_state in self.accept_states
        # steps.append({'type': 'final', 'content': f"Final state: {current_state}"})

        return accepted, steps, state_sequence

class NFA:
    def __init__(self):
        self.states = {'q0', 'q1', 'q2'}
        self.alphabet = {'a', 'b'}
        self.start_state = 'q0'
        self.accept_states = {'q2'}
        self.transitions = {
            'q0': {'a': {'q0', 'q1'}, 'b': {'q0'}},
            'q1': {'a': set(), 'b': {'q2'}},
            'q2': {'a': set(), 'b': set()}
        }
        self.positions = {
            'q0': (100, 150),
            'q1': (250, 150),
            'q2': (400, 150)
        }

    def get_description(self):
        return "NFA: L = {w ∈ {a, b}* | w ends with 'ab'}"

    def generate_random(self):
        length = random.randint(2, 6)
        base = ''.join(random.choice(['a', 'b']) for _ in range(length - 2))
        if random.random() < 0.7:
            return base + 'ab'
        else:
            return base + random.choice(['aa', 'ba'])

    def process(self, input_string):
        current_states = {self.start_state}
        steps = []
        state_sequence = [current_states.copy()]
        
        steps.append({'type': 'initial', 'content': f"Initial states: {{{', '.join(sorted(current_states))}}}"})

        for i, symbol in enumerate(input_string):
            if symbol not in self.alphabet:
                steps.append({'type': 'error', 'content': f"Error: Invalid symbol '{symbol}'"})
                return False, steps, state_sequence

            next_states = set()
            for state in current_states:
                next_states.update(self.transitions[state].get(symbol, set()))

            steps.append({
                'type': 'step',
                'number': i + 1,
                'symbol': symbol,
                'current_states': current_states.copy(),
                'next_states': next_states.copy()
            })
            current_states = next_states
            state_sequence.append(current_states.copy())

            if not current_states:
                steps.append({'type': 'error', 'content': 'No active states'})
                return False, steps, state_sequence

        accepted = any(state in self.accept_states for state in current_states)
        steps.append({'type': 'final', 'content': f"Final states: {{{', '.join(sorted(current_states))}}}"})

        return accepted, steps, state_sequence

class NFAEpsilon:
    def __init__(self):
        self.states = {'q0', 'q1', 'q2', 'q3'}
        self.alphabet = {'a', 'b'}
        self.start_state = 'q0'
        self.accept_states = {'q3'}
        self.transitions = {
            'q0': {'a': {'q0', 'q1'}, 'b': {'q0'}},
            'q1': {'a': set(), 'b': {'q2'}},
            'q2': {'a': set(), 'b': {'q3'}},
            'q3': {'a': set(), 'b': set()}
        }
        self.positions = {
            'q0': (80, 150),
            'q1': (200, 150),
            'q2': (320, 150),
            'q3': (440, 150)
        }

    def get_description(self):
        return "NFA-ε: L = (a|b)*abb"

    def generate_random(self):
        length = random.randint(1, 4)
        base = ''.join(random.choice(['a', 'b']) for _ in range(length))
        if random.random() < 0.7:
            return base + 'abb'
        return base

    def process(self, input_string):
        current_states = {self.start_state}
        steps = []
        state_sequence = [current_states.copy()]
        
        steps.append({'type': 'initial', 'content': f"Initial states: {{{', '.join(sorted(current_states))}}}"})

        for i, symbol in enumerate(input_string):
            if symbol not in self.alphabet:
                steps.append({'type': 'error', 'content': f"Error: Invalid symbol '{symbol}'"})
                return False, steps, state_sequence

            next_states = set()
            for state in current_states:
                next_states.update(self.transitions[state].get(symbol, set()))

            steps.append({
                'type': 'step',
                'number': i + 1,
                'symbol': symbol,
                'current_states': current_states.copy(),
                'next_states': next_states.copy()
            })
            current_states = next_states
            state_sequence.append(current_states.copy())

            if not current_states:
                steps.append({'type': 'error', 'content': 'No active states'})
                return False, steps, state_sequence

        accepted = any(state in self.accept_states for state in current_states)
        steps.append({'type': 'final', 'content': f"Final states: {{{', '.join(sorted(current_states))}}}"})

        return accepted, steps, state_sequence

class PDA:
    def __init__(self):
        self.states = {'q0', 'q1', 'q2'}
        self.alphabet = {'a', 'b'}
        self.start_state = 'q0'
        self.accept_states = {'q0', 'q2'}
        self.positions = {
            'q0': (100, 150),
            'q1': (250, 100),
            'q2': (400, 150)
        }

    def get_description(self):
        return "PDA: L = {a^n b^n | n ≥ 0}"

    def generate_random(self):
        n = random.randint(0, 5)
        return 'a' * n + 'b' * n

    def process(self, input_string):
        state = self.start_state
        stack = ['Z']
        steps = []
        state_sequence = [state]
        
        steps.append({'type': 'initial', 'content': f"Initial: State={state}, Stack=[{', '.join(stack)}]"})

        if input_string == '':
            steps.append({'type': 'info', 'content': 'Empty string: n=0'})
            return True, steps, state_sequence

        i = 0
        while i < len(input_string) and input_string[i] == 'a':
            state = 'q1'
            stack.append('A')
            steps.append({
                'type': 'step',
                'number': i + 1,
                'symbol': 'a',
                'state': state,
                'action': 'Push(A)',
                'stack': stack.copy()
            })
            state_sequence.append(state)
            i += 1

        while i < len(input_string) and input_string[i] == 'b':
            if len(stack) <= 1:
                steps.append({'type': 'error', 'content': "Stack empty (more b's than a's)"})
                return False, steps, state_sequence

            state = 'q2'
            popped = stack.pop()
            steps.append({
                'type': 'step',
                'number': i + 1,
                'symbol': 'b',
                'state': state,
                'action': f'Pop({popped})',
                'stack': stack.copy()
            })
            state_sequence.append(state)
            i += 1

        if i < len(input_string):
            steps.append({'type': 'error', 'content': f"Invalid symbol '{input_string[i]}'"})
            return False, steps, state_sequence

        accepted = len(stack) == 1 and state in self.accept_states
        steps.append({'type': 'final', 'content': f"Final: State={state}, Stack=[{', '.join(stack)}]"})

        if not accepted and len(stack) > 1:
            steps.append({'type': 'info', 'content': "More a's than b's"})

        return accepted, steps, state_sequence

class TuringMachine:
    def __init__(self):
        self.start_state = 'q0'
        self.accept_states = {'q_accept'}
        self.states = {'q0', 'q1', 'q2', 'q3', 'q4', 'q4_y', 'q4_z', 'q_accept', 'q_reject'}
        self.positions = {
            'q0': (100, 250),
            'q1': (200, 250),
            'q2': (300, 250),
            'q3': (400, 250),
            'q4': (200, 150),
            'q4_y': (300, 150),
            'q4_z': (400, 150),
            'q_accept': (100, 50),
            'q_reject': (500, 50)
        }

    def get_description(self):
        return "TM: L = {a^n b^n c^n | n ≥ 1}"

    def generate_random(self):
        n = random.randint(1, 3)
        return 'a' * n + 'b' * n + 'c' * n

    def process(self, input_string):
        if input_string == '':
            return False, [{'type': 'error', 'content': 'Empty string (n ≥ 1 required)'}], []

        tape = list(input_string) + ['_'] * 10
        head = 0
        state = self.start_state
        steps = []
        state_sequence = [state]
        
        steps.append({'type': 'initial', 'content': f"Initial tape: {''.join(tape[:len(input_string)])}"})

        step_count = 0
        max_steps = 1000

        while state not in {'q_accept', 'q_reject'} and step_count < max_steps:
            step_count += 1
            current_symbol = tape[head] if head < len(tape) else '_'

            if state == 'q0':
                if current_symbol == 'a':
                    tape[head] = 'X'
                    head += 1
                    state = 'q1'
                    steps.append({'type': 'tm_step', 'number': step_count, 'action': "Read 'a', write 'X', move R", 'state': state, 'head': head})
                elif current_symbol in ['X', 'Y']:
                    head += 1
                else:
                    state = 'q_reject'
                    
            elif state == 'q1':
                if current_symbol == 'a':
                    head += 1
                elif current_symbol == 'Y':
                    head += 1
                elif current_symbol == 'b':
                    tape[head] = 'Y'
                    head += 1
                    state = 'q2'
                    steps.append({'type': 'tm_step', 'number': step_count, 'action': "Read 'b', write 'Y', move R", 'state': state, 'head': head})
                elif current_symbol == 'X':
                    head += 1
                else:
                    state = 'q_reject'
                    
            elif state == 'q2':
                if current_symbol == 'b':
                    head += 1
                elif current_symbol == 'Z':
                    head += 1
                elif current_symbol == 'c':
                    tape[head] = 'Z'
                    head -= 1
                    state = 'q3'
                    steps.append({'type': 'tm_step', 'number': step_count, 'action': "Read 'c', write 'Z', move L", 'state': state, 'head': head})
                elif current_symbol == 'Y':
                    head += 1
                else:
                    state = 'q_reject'
                    
            elif state == 'q3':
                if current_symbol in ['a', 'b', 'Y', 'X']:
                    head -= 1
                elif current_symbol == 'Z':
                    head -= 1
                elif current_symbol == '_':
                    head = 0
                    state = 'q4'
                    steps.append({'type': 'tm_step', 'number': step_count, 'action': "Reset head to start, begin final check", 'state': state, 'head': head})
                else:
                    state = 'q_reject'
                    
            elif state == 'q4':
                if current_symbol == 'X':
                    head += 1
                elif current_symbol == 'Y':
                    state = 'q4_y'
                    head += 1
                    steps.append({'type': 'tm_step', 'number': step_count, 'action': "Start Y sweep", 'state': state, 'head': head})
                elif current_symbol == '_':
                    state = 'q_accept'
                else:
                    state = 'q_reject'
                    
            elif state == 'q4_y':
                if current_symbol == 'Y':
                    head += 1
                elif current_symbol == 'Z':
                    state = 'q4_z'
                    head += 1
                    steps.append({'type': 'tm_step', 'number': step_count, 'action': "Start Z sweep", 'state': state, 'head': head})
                else:
                    state = 'q_reject'
                    
            elif state == 'q4_z':
                if current_symbol == 'Z':
                    head += 1
                elif current_symbol == '_':
                    state = 'q_accept'
                    steps.append({'type': 'tm_step', 'number': step_count, 'action': "All symbols verified - ACCEPT", 'state': state, 'head': head})
                else:
                    state = 'q_reject'

            state_sequence.append(state)

        if step_count >= max_steps:
            state = 'q_reject'
            steps.append({'type': 'error', 'content': 'Maximum steps exceeded'})

        accepted = state == 'q_accept'
        steps.append({'type': 'final', 'content': f"Final tape: {''.join(tape[:max(20, len(input_string))])}"})
        steps.append({'type': 'info', 'content': f"Total steps: {step_count}"})
        # steps.append({'type': 'info', 'content': f"Final state: {state}"})

        return accepted, steps, state_sequence

def create_state_diagram(automaton_type, active_states=None):
    """Create state diagram visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    automata_classes = {
        'DFA': DFA,
        'NFA': NFA,
        'NFAe': NFAEpsilon,
        'PDA': PDA,
        'TM': TuringMachine
    }
    
    automaton = automata_classes[automaton_type]()
    
    G = nx.DiGraph()
    
    # Define transitions for visualization
    if automaton_type == 'DFA':
        G.add_nodes_from(['q0', 'q1'])
        G.add_edge('q0', 'q1', label='0')
        G.add_edge('q0', 'q0', label='1')
        G.add_edge('q1', 'q0', label='0')
        G.add_edge('q1', 'q1', label='1')
        title = "DFA - Even zeros"
        
    elif automaton_type == 'NFA':
        G.add_nodes_from(['q0', 'q1', 'q2'])
        G.add_edge('q0', 'q0', label='a, b')
        G.add_edge('q0', 'q1', label='a')
        G.add_edge('q1', 'q2', label='b')
        title = "NFA - Ends 'ab'"
        
    elif automaton_type == 'NFAe':
        G.add_nodes_from(['q0', 'q1', 'q2', 'q3'])
        G.add_edge('q0', 'q1', label='a')
        G.add_edge('q0', 'q0', label='a, b')
        G.add_edge('q1', 'q2', label='b')
        G.add_edge('q2', 'q3', label='b')
        title = "NFA-ε - (a|b)*abb"
        
    elif automaton_type == 'PDA':
        G.add_nodes_from(['q0', 'q1', 'q2'])
        G.add_edge('q0', 'q1', label='a, ε→A')
        G.add_edge('q1', 'q1', label='a, ε→A')
        G.add_edge('q1', 'q2', label='b, A→ε')
        G.add_edge('q2', 'q2', label='b, A→ε')
        title = "PDA - aⁿbⁿ"
        
    elif automaton_type == 'TM':
        G.add_nodes_from(['q0', 'q1', 'q2', 'q3', 'q4', 'q4_y', 'q4_z', 'q_accept', 'q_reject'])
        G.add_edge('q0', 'q1', label='a→X,R')
        G.add_edge('q1', 'q1', label='a→a,R\nY→Y,R')
        G.add_edge('q1', 'q2', label='b→Y,R')
        G.add_edge('q2', 'q2', label='b→b,R\nZ→Z,R')
        G.add_edge('q2', 'q3', label='c→Z,L')
        G.add_edge('q3', 'q3', label='a,b,Y,Z→L')
        G.add_edge('q3', 'q4', label='_→_,R\n(reset)')
        G.add_edge('q4', 'q4', label='X→R')
        G.add_edge('q4', 'q4_y', label='Y→R')
        G.add_edge('q4_y', 'q4_y', label='Y→R')
        G.add_edge('q4_y', 'q4_z', label='Z→R')
        G.add_edge('q4_z', 'q4_z', label='Z→R')
        G.add_edge('q4_z', 'q_accept', label='_→_,R')
        title = "TM - aⁿbⁿcⁿ"
    
    # Use predefined positions
    pos = automaton.positions
    
    # State highlighting logic - FIXED
    node_colors = []
    node_borders = []
    
    for node in G.nodes():
        # Determine if this node should be highlighted as active
        is_active = False
        if active_states is not None:
            # Handle both set (NFA) and string (DFA) active states
            if isinstance(active_states, set):
                is_active = node in active_states
            elif isinstance(active_states, list):
                is_active = node in active_states
            else:
                # Single state (string)
                is_active = str(node) == str(active_states)
        
        # Determine node color based on state type and activity
        # Priority: Active > Accept > Start > Regular
        if is_active:
            node_colors.append('yellow')
            node_borders.append('red')
        elif node in automaton.accept_states:
            node_colors.append('lightgreen')
            node_borders.append('darkgreen')
        elif node == automaton.start_state:
            node_colors.append('lightblue')
            node_borders.append('darkblue')
        else:
            node_colors.append('white')
            node_borders.append('gray')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=2000, edgecolors=node_borders, 
                          linewidths=4, ax=ax)
    
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20, 
                          edge_color='gray', width=2, ax=ax,
                          connectionstyle="arc3,rad=0.1")
    
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                               font_size=8, bbox=dict(facecolor='white', alpha=0.7), ax=ax)
    
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                   markeredgecolor='darkblue', markeredgewidth=2, markersize=10, label='Start State'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                   markeredgecolor='darkgreen', markeredgewidth=2, markersize=10, label='Accept State'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', 
                   markeredgecolor='red', markeredgewidth=2, markersize=10, label='Active State'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
                   markeredgecolor='gray', markeredgewidth=2, markersize=10, label='Regular State')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1), framealpha=0.9)
    
    # Set appropriate limits
    if automaton_type == 'TM':
        ax.set_xlim(-50, 600)
        ax.set_ylim(-50, 350)
    else:
        ax.set_xlim(-50, 550)
        ax.set_ylim(50, 250)
        
    ax.axis('off')
    plt.tight_layout()
    
    # Convert to base64 for displaying in Streamlit
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)  # Properly close the figure
    
    return f"data:image/png;base64,{img_str}"

def main():
    # Initialize session state
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []
    if 'current_automaton' not in st.session_state:
        st.session_state.current_automaton = "DFA"
    if 'current_active_states' not in st.session_state:
        st.session_state.current_active_states = None
    
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; border-radius: 10px; color: white; text-align: center; margin-bottom: 30px;">
        <h1 style="margin: 0; font-size: 2.5em;">🤖 Computational Automata Simulator</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Automaton Selection
        st.markdown("### Select Automaton")
        
        automaton_options = {
            "DFA": "Even zeros",
            "NFA": "Ends 'ab'", 
            "NFAe": "(a|b)*abb",
            "PDA": "aⁿ bⁿ",
            "TM": "aⁿ bⁿ cⁿ"
        }
        
        selected_automaton = st.radio(
            "Choose automaton type:",
            options=list(automaton_options.keys()),
            format_func=lambda x: f"{x} - {automaton_options[x]}",
            key="automaton_selector"
        )
        
        # Reset highlighting when automaton changes
        if selected_automaton != st.session_state.current_automaton:
            st.session_state.current_automaton = selected_automaton
            st.session_state.current_active_states = None
            
            # Add to history
            automata_classes = {
                'DFA': DFA,
                'NFA': NFA,
                'NFAe': NFAEpsilon,
                'PDA': PDA,
                'TM': TuringMachine
            }
            description = automata_classes[selected_automaton]().get_description()
            st.session_state.processing_history.insert(0, {
                'type': 'info',
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'content': f"📌 Switched to: {description}",
                'automaton': selected_automaton
            })
        
        # Input Section
        st.markdown("---")
        st.markdown("### Input String")
        
        input_string = st.text_input(
            "Enter input string here...",
            placeholder="e.g., '0011' for DFA, 'aabb' for PDA...",
            key="input_string"
        )
        
        # Process Buttons
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            if st.button("🚀 Process", use_container_width=True):
                if input_string:
                    process_input(selected_automaton, input_string)
                else:
                    st.warning("⚠️ Please enter an input string")
        
        with col1_2:
            if st.button("🎲 Random Tests", use_container_width=True):
                run_random_tests(selected_automaton)
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.processing_history = []
            st.session_state.current_active_states = None
            st.rerun()
        
        # Processing Steps & History
        st.markdown("---")
        st.markdown("### Processing Steps & History")
        
        if not st.session_state.processing_history:
            st.info("""
            👆 Welcome to the Automata Simulator
            
            Select an automaton and enter input to see step-by-step execution
            Click "Random Tests" to generate and test random valid inputs
            """)
        else:
            history_container = st.container()
            with history_container:
                for item in st.session_state.processing_history:
                    if item['type'] == 'execution':
                        with st.expander(f"[{item['timestamp']}] {item['automaton']} | Input: '{item['input']}' → {'✅ ACCEPTED' if item['accepted'] else '❌ REJECTED'}", expanded=True):
                            display_execution_steps(item)
                    elif item['type'] == 'info':
                        st.info(item['content'])
                    elif item['type'] == 'test_header':
                        st.success(item['content'])
    
    with col2:
        # State Diagram
        st.markdown("### State Diagram")
        
        diagram_img = create_state_diagram(
            st.session_state.current_automaton,
            st.session_state.current_active_states
        )
        
        st.markdown(
            f'<img src="{diagram_img}" style="width: 100%; border: 2px solid #ddd; border-radius: 10px;">',
            unsafe_allow_html=True
        )
        
        # Show current highlighting status
        if st.session_state.current_active_states:
            st.markdown(f"**Active State:** `{st.session_state.current_active_states}`")
        
        # Current Automaton Info
        st.markdown("---")
        st.markdown("### Current Automaton")
        
        automata_classes = {
            'DFA': DFA,
            'NFA': NFA,
            'NFAe': NFAEpsilon,
            'PDA': PDA,
            'TM': TuringMachine
        }
        
        automaton = automata_classes[st.session_state.current_automaton]()
        st.info(f"**{automaton.get_description()}**")

def process_input(automaton_type, input_string):
    """Process input string with selected automaton"""
    automata_classes = {
        'DFA': DFA,
        'NFA': NFA,
        'NFAe': NFAEpsilon,
        'PDA': PDA,
        'TM': TuringMachine
    }
    
    automaton = automata_classes[automaton_type]()
    accepted, steps, state_sequence = automaton.process(input_string)
    
    # Add to history FIRST
    st.session_state.processing_history.insert(0, {
        'type': 'execution',
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'automaton': automaton_type,
        'input': input_string,
        'accepted': accepted,
        'steps': steps,
        'state_sequence': state_sequence
    })
    
    # Set the active state for highlighting AFTER adding to history
    if state_sequence and len(state_sequence) > 0:
        final_state = state_sequence[-1]
        st.session_state.current_active_states = final_state
    else:
        st.session_state.current_active_states = None
    
    st.rerun()

def display_execution_steps(execution):
    """Display execution steps in a formatted way"""
    steps = execution['steps']
    automaton_type = execution['automaton']
    
    # Show state sequence info
    if 'state_sequence' in execution and execution['state_sequence']:
        seq_str = str(execution['state_sequence'])
        final_state = execution['state_sequence'][-1]
        
        st.markdown(f"**State Sequence:** `{seq_str}`")
        st.markdown(f"**Final State:** `{final_state}`")
        st.markdown(f"**Sequence Length:** `{len(execution['state_sequence'])}`")
        
        # Show if THIS execution's final state is currently being highlighted in the diagram
        is_currently_highlighted = (
            hasattr(st.session_state, 'current_active_states') and 
            st.session_state.current_active_states == final_state
        )
        # if is_currently_highlighted:
        #     st.markdown(f"**🔦 This state is currently highlighted in the diagram**")
    
    # Create a table for steps
    table_data = []
    
    for step in steps:
        if step['type'] == 'initial':
            st.markdown(f"**Initial:** {step['content']}")
        elif step['type'] == 'final':
            st.markdown(f"**Final:** {step['content']}")
        elif step['type'] == 'error':
            st.error(step['content'])
        elif step['type'] == 'info':
            st.info(step['content'])
        elif step['type'] == 'step':
            if automaton_type == 'DFA':
                table_data.append({
                    'Step': step['number'],
                    'Current State': step['current_state'],
                    'Read': f"'{step['symbol']}'",
                    'Next State': step['next_state']
                })
            elif automaton_type in ['NFA', 'NFAe']:
                curr_states = ', '.join(sorted(step['current_states'])) if step['current_states'] else '∅'
                next_states = ', '.join(sorted(step['next_states'])) if step['next_states'] else '∅'
                table_data.append({
                    'Step': step['number'],
                    'Read': f"'{step['symbol']}'",
                    'Current States': f"{{{curr_states}}}",
                    'Next States': f"{{{next_states}}}"
                })
            elif automaton_type == 'PDA':
                table_data.append({
                    'Step': step['number'],
                    'Read': f"'{step['symbol']}'",
                    'State': step['state'],
                    'Action': step['action'],
                    'Stack': f"[{', '.join(step['stack'])}]"
                })
        elif step['type'] == 'tm_step':
            table_data.append({
                'Step': step.get('number', 'N/A'),
                'State': step.get('state', 'N/A'),
                'Action': step.get('action', 'N/A'),
                'Head Position': step.get('head', 'N/A')
            })
    
    if table_data:
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

def run_random_tests(automaton_type):
    """Run random test cases"""
    automata_classes = {
        'DFA': DFA,
        'NFA': NFA,
        'NFAe': NFAEpsilon,
        'PDA': PDA,
        'TM': TuringMachine
    }
    
    automaton = automata_classes[automaton_type]()
    num_tests = 5
    
    # Add test header to history
    st.session_state.processing_history.insert(0, {
        'type': 'test_header',
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'content': f"🎲 GENERATING RANDOM TEST CASE FOR {automaton_type}"
    })
    
    # Generate and process test cases
    test_cases = [automaton.generate_random() for _ in range(num_tests)]
    
    for test_case in test_cases:
        process_input(automaton_type, test_case)
        time.sleep(0.5)
    
    # Add completion message
    st.session_state.processing_history.insert(0, {
        'type': 'info',
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'content': f"✅ Completed {len(test_cases)} random test cases for {automaton_type}"
    })
    
    st.rerun()

if __name__ == "__main__":
    main()