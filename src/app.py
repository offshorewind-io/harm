import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import numpy as np
import plotly.express as px
import pandas as pd


def macaulay(x):
    x[x<0] = 0
    return x


def calculate_harm(monotonic_loading_parameters, ratcheting_parameters, modelling_parameters):
    
    N_s = modelling_parameters['N_s']
    k_u = monotonic_loading_parameters['k_u']
    epsilon_pu = monotonic_loading_parameters['epsilon_pu']
    m_h = monotonic_loading_parameters['m_h']

    H_0 = monotonic_loading_parameters['E_0']
    n = np.arange(1, N_s + 1)
    H_i = k_u / (N_s * epsilon_pu) * N_s**m_h / ((n + 1)**m_h - 2 * n**m_h + (n - 1)**m_h)
    k_i = k_u * n / N_s

    sigma_max = modelling_parameters['sigma_max']
    sigma_min = modelling_parameters['sigma_min']
    d_t = modelling_parameters['d_t']
    mu = modelling_parameters['mu']

    n_step = np.round(modelling_parameters['T'] / 2 / d_t).astype(int)
    sigma_load_initial = np.linspace(0, sigma_max, n_step)
    sigma_load_down = np.linspace(sigma_max, sigma_min, n_step)[1:-1]
    sigma_load_up = np.linspace(sigma_min, sigma_max, n_step)
    sigma_load_cycle = list(sigma_load_down) + list(sigma_load_up)
    sigma_load = np.array(list(sigma_load_initial) + sigma_load_cycle * modelling_parameters['n_cycles'])

    R_0 = ratcheting_parameters['R_0']
    m_r = ratcheting_parameters['m_r']
    m_s = ratcheting_parameters['m_s']
    beta_0 = ratcheting_parameters['beta_0']

    sigma_0 = 0
    epsilon_0 = 0
    sigma_output = [0]
    epsilon_output = [0]
    alpha_i = np.zeros(len(k_i))
    alpha_r = 0
    beta = beta_0 * 1

    d_output = np.round(len(sigma_load) / 3000).astype(int)

    for i, sigma in enumerate(sigma_load[1:]):

        d_sigma = sigma - sigma_0
        d_alpha_i = d_t / mu * macaulay(np.abs(sigma - H_i * alpha_i) - k_i) * np.sign(sigma - H_i * alpha_i) 

        R_i = R_0 * (k_i / k_u) * (beta / beta_0)**(-m_r) * (np.abs(sigma) / k_u)**m_s
        d_alpha_r = np.sign(sigma) * np.sum(R_i * np.abs(d_alpha_i))
        d_epsilon = d_sigma / H_0 + np.sum(d_alpha_i) + d_alpha_r

        alpha_i = alpha_i + d_alpha_i
        alpha_r = alpha_r + d_alpha_r
        beta = beta + np.abs(d_alpha_r)
        epsilon = epsilon_0 + d_epsilon
        sigma_0 = sigma * 1
        epsilon_0 = epsilon * 1

        if i % d_output == 0:
            epsilon_output = epsilon_output + [epsilon]
            sigma_output = sigma_output + [sigma]

    return sigma_output, epsilon_output


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
#server = app.server

def input_div(name, id, default_value):

    col = dbc.Col([
        dbc.Label(name),
        dbc.Input(type="number", id=id, value=default_value)],
        md=4
    )

    return col


accordion = html.Div(
    dbc.Accordion([
        dbc.AccordionItem([
                dbc.Row([
                    input_div('E_0', 'E_0', 85),
                    input_div('k_u', 'k_u', 1.0),
                    input_div('epsilon_pu', 'epsilon_pu', 1.0),
                    input_div('m_h', 'm_h', 3.2),
                ])
                ],
                title="Monotonic loading parameters",
            ),
            dbc.AccordionItem([
                dbc.Row([
                    input_div('R_0', 'R_0', 4000),
                    input_div('beta_0', 'beta_0', 0.05),
                    input_div('m_r', 'm_r', 4.5),
                    input_div('m_s', 'm_s', 5.0),
                ])
                ],
                title="Ratcheting parameters",
            ),
            dbc.AccordionItem([
                dbc.Row([
                    input_div('N_s', 'N_s', 40),
                    input_div('mu', 'mu', 0.1),
                    input_div('d_t', 'd_t', 0.0033),
                    input_div('T', 'T', 10),
                    input_div('sigma_max', 'sigma_max', 0.51),
                    input_div('sigma_min', 'sigma_min', 0.08),
                    input_div('n_cycles', 'n_cycles', 10),
                ])
                ],
                title="Modelling parameters",
            ),
        ],
    ),
)


app.layout = html.Div([
    html.H1("HARM - Hyperplastic Accelerated Ratcheting Model"),
        html.H3("G.T. Houlsby, C.A. Abadie, W.J.A.P. Beuckelaers, B.W. Byrne"),
    dbc.Row([
        dbc.Col([
            accordion,
            dbc.Button("Calculate", id='calculate', color="primary"),
        ], md=4, className="d-grid"),
        dbc.Col([
            dcc.Graph(id='fig')
        ], md=8)
    ], style={"margin-top": "40px"})
], style={"margin": "15px"})


@app.callback([Output('fig', 'figure')], [Input('calculate', 'n_clicks')], [State('E_0', 'value'), State('k_u', 'value'), State('epsilon_pu', 'value'),
State('m_h', 'value'), State('R_0', 'value'), State('beta_0', 'value'), State('m_r', 'value'), State('m_s', 'value'),
State('N_s', 'value'), State('mu', 'value'), State('d_t', 'value'), State('T', 'value'), State('sigma_max', 'value'), State('sigma_min', 'value'),
State('n_cycles', 'value')])
def update_figure(button, E_0, k_u, epsilon_pu, m_h, R_0, beta_0, m_r, m_s, N_s, mu, d_t, T, sigma_max, sigma_min, n_cycles):

    monotonic_loading_parameters = {'E_0': E_0, 'k_u': k_u, 'epsilon_pu': epsilon_pu, 'm_h': m_h}
    ratcheting_parameters = {'R_0': R_0, 'beta_0': beta_0, 'm_r': m_r, 'm_s': m_s, 'm_k': 0.0}
    modelling_parameters = {'N_s': N_s, 'mu': mu, 'd_t': d_t, 'T': T, 'sigma_max': sigma_max, 'sigma_min': sigma_min, 'n_cycles': n_cycles}

    sigma_load, epsilon = calculate_harm(monotonic_loading_parameters, ratcheting_parameters, modelling_parameters)

    df = pd.DataFrame(data={'epsilon': epsilon, 'sigma': sigma_load})
    fig = px.line(df, x='epsilon', y='sigma')

    return (fig,)



if __name__ == '__main__':
    app.run_server(debug=True, port=8080)  
