# stdlib
import ast
import concurrent.futures
import copy
import threading
import time
import traceback

# 3rd party
import bqplot as bq
import ipyparallel as ipp
from IPython.display import display, clear_output, update_display
import ipywidgets as ipw
import numpy as np
import pandas as pd
import qgrid

from kale.services.worker import KaleWorkerClient
from kale.services.manager import KaleManagerClient
from kale.widgets import KaleWorkerResourcesBoard

class ModelPlot(ipw.VBox):
    def __init__(self, y, x=None, xlim=None, ylim=None, xlabel=None, ylabel=None, title=None):
        super().__init__()

        self.x = x
        self.xlim = xlim or [0, 1]
        self.ylim = ylim or [0, 1]
        self.xlabel = xlabel or 'x'
        self.ylabel = ylabel or 'y'
        self.title = title or "{} vs {}".format(self.ylabel, self.xlabel)

        if isinstance(y, list):
            self.y = y
        else:
            self.y = [y]

        self.colors = ['blue', 'red', 'green', 'orange', 'black', 'purple', 'gray']
        self.xscale = bq.LinearScale(min=self.xlim[0], max=self.xlim[1])
        self.yscale = bq.LinearScale(min=self.ylim[0], max=self.ylim[1])

        if isinstance(self.ylabel, list):
            ylabel = ''
        else:
            ylabel = self.ylabel
            
        self.xax = bq.Axis(
            scale=self.xscale,
            label=self.xlabel,
            grid_lines='none',
        )
        self.yax = bq.Axis(
            scale=self.yscale,
            label=ylabel,
            orientation='vertical',
            grid_lines='none',
        )
        self.num_lines = 0
        self.lines = []
        self.scatters = []
        self.labels = []
        
        if isinstance(self.y, list):
            for y in self.y:
                self.create_line(y, display_legend=True)
        else:
            self.create_line(self.y)

        self.fig = bq.Figure(
            marks=self.lines + self.scatters, 
            axes=[self.xax, self.yax], 
            layout=ipw.Layout(height='500px', width='100%', overflow_x='hidden'),
            title=self.title)
        self.debug = ipw.Output(layout=ipw.Layout(height='100px', overflow_y='scroll'))
        self.children = [self.fig]

    def create_line(self, y, display_legend=False):
        try:
            color = self.colors[self.num_lines % len(self.colors)]
            self.lines.append(bq.Lines(
                x=[],
                y=[],
                scales={'x': self.xscale, 'y': self.yscale},
                interpolation='linear',
                display_legend=display_legend,
                colors=[color],
                labels=[y],
                enable_hover=True
            ))
            self.scatters.append(bq.Scatter(
                x=[],
                y=[],
                scales={'x': self.xscale, 'y': self.yscale},
                colors=[color],
                enable_hover=True
            ))
            self.labels.append(y)
            self.num_lines += 1

            self.lines[-1].tooltip = bq.Tooltip(
                fields=['name'],
                show_labels=True)
            self.lines[-1].interactions = {
                'hover': 'tooltip',
                'click': 'tooltip'
            }
                        
            self.scatters[-1].tooltip = bq.Tooltip(
                fields=['y','x'],
                labels=[y, self.xlabel], 
                formats=['.4f', ''],
                show_labels=True)
            self.scatters[-1].interactions = {
                'hover': 'tooltip',
                'click': 'tooltip'
            }
        except Exception as e:
            self.debug.append_stdout("Exception when adding a line and points to plot: {}".format(e.args))

    def resize_fig(self):
        try:
            for i in range(len(self.lines)):
                if len(self.lines[i].x) > 0:
                    self.xscale.min = min(self.xscale.min, float(np.min(self.lines[i].x)))
                    self.xscale.max = max(self.xscale.max, float(np.max(self.lines[i].x)))
                    self.yscale.min = min(self.yscale.min, float(np.min(self.lines[i].y)))
                    self.yscale.max = max(self.yscale.max, float(np.max(self.lines[i].y)))
        except Exception as e:
            self.debug.append_stdout("Exception when resizing the figure: {}\n".format(e.args))

    def update(self, data):
        try:
            for i in range(self.num_lines):
                self.lines[i].y = np.array(data[self.y[i]])
                self.scatters[i].y = np.array(data[self.y[i]])

                if self.x and self.x in data:
                    self.lines[i].x = np.array(data[self.x])
                    self.scatters[i].x = np.array(data[self.x])
                else:
                    self.lines[i].x = np.array([i for i in range(len(self.lines[0].y))])
                    self.scatters[i].x = np.array([i for i in range(len(self.lines[0].y))])
                
            self.resize_fig()
        except Exception as e:
            self.debug.append_stdout("Exception while plotting lines and resizing figure: {}\n".format(e.args))
            self.debug.append_stdout("Data: {}\n".format(data))


class ModelParamEditor(ipw.HBox):
    def __init__(self, callback, params, layout=None):
        super().__init__()
        
        self._callback = callback
        self._params = params
        self._layout = None
        self._inputs = {}
        
        self._model_id = ipw.Label(value="Model {}".format(0))
        
        for k in self._params:
            if "options" in self._params[k]:
                self._inputs[k] = ipw.Dropdown(
                    description = k, 
                    disabled = False, 
                    options = self._params[k]["options"], 
                    value = self._params[k]["default"])
                continue
            
            if self._params[k]["type"] == int:
                self._inputs[k] = ipw.IntText(description=k, disabled=False, value=self._params[k]["default"])
            elif self._params[k]["type"] == float:
                self._inputs[k] = ipw.FloatText(description=k, disabled=False, value=self._params[k]["default"])
            elif self._params[k]["type"] == bool:
                self._inputs[k] = ipw.Checkbox(description=k, disabled=False, value=self._params[k]["default"])
            else:
                self._inputs[k] = ipw.Text(description=k, disabled=False, value=str(self._params[k]["default"]))
        
        _children = [ipw.HBox([self._model_id])]
        param_display_row = []
        for k,v in self._inputs.items():
            param_display_row.append(v)
            if len(param_display_row) == 3:
                _children.append(ipw.HBox(param_display_row))
                param_display_row = []
        if len(param_display_row) > 0:
            _children.append(ipw.HBox(param_display_row))
        
        self.children = [ipw.VBox(_children)]
        self.layout.display = 'none'
    
    def display(self, row_id, model_id, values):
        self._model_id.value = "Model {}".format(model_id)
        for k in values:
            if isinstance(values[k], int) or isinstance(values[k], float) or isinstance(values[k], bool):
                self._inputs[k].value = values[k]
            else:
                self._inputs[k].value = str(values[k])

    def toggle_disabled(self):
        self._model_id.disabled = not self._model_id.disabled
        for k in self._inputs:
            self._inputs[k].disabled = not self._inputs[k].disabled

    def get_values(self):
        vals = {}
        for k in self._inputs:
            if self._params[k]["type"] == list:
                vals[k] = ast.literal_eval(self._inputs[k].value)
            else:
                vals[k] = self._inputs[k].value
        return vals


class ParamSpanWidget(ipw.VBox):
    def __init__(self, compute_func, vis_func, params, columns=None, ipp_cluster_id=None, 
                 output_layout=None, qgrid_layout=None, kale_manager=None):
        """
        compute_func: function 
        task to submit to IPyParallel for model output
        
        vis_func: function 
        function that produces a visualization of the model output (e.g. ModelPlot)

        params: dict
        grid search parameters, either lists/numpy arrays or list of lists/2D numpy arrays, 
        where the outer lists have the same length
        
        ipp_cluster_id: str
        optional ipyparallel cluster id for connecting to a specific controller
        """
        super().__init__()

        self.compute_func = compute_func
        self.vis_func = vis_func
        self.output_layout = output_layout or \
            ipw.Layout(height='500px', 
                       border='1px solid', overflow_x='hidden', overflow_y='scroll')
        self.debug_layout = ipw.Layout(height='500px', border='1px solid', overflow_x='scroll', overflow_y='scroll')
        self.qgrid_layout = qgrid_layout or \
            ipw.Layout(height='500px', overflow_y='scroll')
        self.layout = ipw.Layout(height='auto', min_height='1050px', max_height='3000px', overflow_y='scroll')

        list_params = {}
        for k in params:
            if type(params[k]["values"]) is np.ndarray:
                list_params[k] = params[k]["values"].tolist()
            else:
                list_params[k] = list(params[k]["values"])

        self.compute_params = list_params
        self.columns = ["status", "epoch"] + [k for k in params] + ["loss", "val_loss", "acc", "val_acc"]
        
        self._param_definitions = {}
        for k in params:
            self._param_definitions[k] = {}
            
            if "default" in params[k]:
                self._param_definitions[k]["default"] = params[k]["default"]
            else:
                self._param_definitions[k]["default"] = list_params[k][0]
                
            self._param_definitions[k]["type"] = params[k]["type"]
            
            if "options" in params[k]:
                self._param_definitions[k]["options"] = params[k]["options"]

        display_params = copy.deepcopy(list_params)
        for k in display_params:
            needs_str = False
            for i in range(len(display_params[k])):
                if isinstance(list, type(display_params[k][i])):
                    needs_str = True
            if needs_str:
                display_params[k] = [str(i) for i in display_params[k]]
        
        # setup the dataframe used to populate the table
        #self.compute_param_keys = params.keys()
        self.params_df = pd.DataFrame(display_params, columns=self.columns)
        self.params_df["status"] = ["Not Started"] * self.params_df.shape[0]
        self.params_df["epoch"] = [-1] * self.params_df.shape[0]

        # create the model plot, resources, and debug output widgets
        self.plot_output = ipw.Output(layout=self.output_layout)
        self.resources_output = ipw.Output(layout=self.output_layout)
        self.debug = ipw.Output(layout=self.debug_layout)

        self.debug.append_stdout("compute_params: {}".format(self.compute_params))

        self._grid_options = {
            "defaultColumnWidth": 200,
            "forceFitColumns": True,
            "editable": False,
            "minVisibleRows": 10,
            "maxVisibleRows": 30
        }
        
        # create the table widget
        self.param_table = qgrid.QGridWidget(
            df=self.params_df, 
            grid_options=self._grid_options,
            layout=self.qgrid_layout)

        # add event listeners to the table
        self.add_handlers()

        # add buttons for stopping and restarting runs
        self._stop_btn = ipw.Button(description="Stop")
        self._stop_btn.on_click(self.stop_selected_models)
        self._restart_btn = ipw.Button(description="Restart")
        self._restart_btn.on_click(self.restart_selected_models)
        self._edit_selected_rows = ModelParamEditor(
            self.update_row, 
            self._param_definitions,
            layout=ipw.Layout(height='300px', border='1px solid', overflow_x='scroll', overflow_y='scroll'))
        self.controls = ipw.VBox([
            ipw.HBox([self._stop_btn, self._restart_btn], layout=ipw.Layout(height='50px')),
            ipw.HBox([self._edit_selected_rows])
        ])
        self.controls.layout.display = 'none'
        self._restart_btn.disabled = True
        self._stop_btn.disabled = True
        
        self._model_tabs = ipw.Tab([self.plot_output, self.resources_output])
        self._model_tabs.set_title(0, "Model Results")
        self._model_tabs.set_title(1, "Resource Usage")
        
        # Add the widgets to this container
        self.children = [
            self._model_tabs, 
            self.controls, 
            self.param_table
        ]
        
        # store all the model related elements and futures
        self._num_models = self.param_table.get_changed_df().shape[0]
        self.model_plots = [self.vis_func(title="Model {}: {}".format(i, 
                {k: self.compute_params[k][i] for k in self.compute_params})) for i in range(self._num_models)]
        self.model_resource_plots = [KaleWorkerResourcesBoard() for i in range(self._num_models)]
        self.model_displays = [None for i in range(self._num_models)]
        self.model_resource_displays = [None for i in range(self._num_models)]
        self.model_data = [
            ModelTaskData(["epoch","loss","val_loss","acc","val_acc"],["status","epoch"]) for i in range(self._num_models)]
        self._model_controller = ModelController(ipp_cluster_id=ipp_cluster_id, kale_manager=kale_manager)

        # select the first row by default
        self._active_plot = 0
        self.param_table._handle_qgrid_msg_helper({'type': 'selection_changed', 'rows': [0]})
        self.param_table._rebuild_widget()
        
        self._stop_updates = threading.Event()
        self._stop_updates.clear()
        self._update_thread = threading.Thread(target=self.update_data)
        self._update_thread.start()

    def add_handlers(self):
        """Add event handlers to the table"""
        self.param_table.on('selection_changed', self.display_selected)
        self.param_table.on('All', self._debug_events)

    def remove_handlers(self):
        """Remove event handlers from the table"""
        self.param_table.off('selection_changed', self.display_selected)
        self.param_table.off('All', self._debug_events)
        
    def submit_computations(self):
        """Start all models"""
        try:            
            for i in range(self._num_models):
                self._model_controller.start_model(
                    i,
                    self.compute_func, 
                    {k: self.compute_params[k][i] for k in self.compute_params})
        except Exception as e:
            self.debug.append_stdout("Exception while submitting runs: {} - {}\n".format(
                e.args, 
                traceback.format_exception(etype=e.__class__, value=e, tb=e.__traceback__)))

    def update_data(self, interval=1):
        try:
            while not self._stop_updates.is_set():
                active_models = self._model_controller.get_running_models()

                for model_id in active_models:
                    data = active_models[model_id].data
                    latest_values = self.param_table.get_changed_df()
                    updated_values = {}
                    
                    if len(data) == 0:
                        continue
                    
                    if "history" in data and len(data["history"]["epoch"]) > 0:
                        current_data_length = self.model_data[model_id].num_data_rows
                        history_data_length = len(data["history"]["epoch"])

                        if current_data_length < history_data_length:
                            if current_data_length == 0:
                                i = 0
                            else:
                                i = current_data_length - 1
                        
                            while i < history_data_length:
                                self.model_data[model_id].append_plot_data_row(
                                    {k: data["history"][k][i] for k in data["history"]})
                                i += 1

                            # apply plot data update
                            if model_id == self._active_plot:
                                self.model_plots[model_id].update(self.model_data[model_id].get_plot_data())

                            for k in data["history"]:
                                updated_values[k] = data["history"][k][-1]
                                
                    if "status" in data and latest_values["status"][model_id] != data["status"]:
                        updated_values["status"] = data["status"]
                                            
                    if "epoch" in data and latest_values["epoch"][model_id] != data["epoch"]:
                        updated_values["epoch"] = data["epoch"]

                    if len(updated_values) > 0:
                        self.update_row(model_id, updated_values)

                        if "status" in updated_values and updated_values["status"] == "Ended Training":
                            self._model_controller.set_model_completed(model_id)
                            
                            if self._active_plot == model_id:
                                self.display_selected({'old': [model_id], 'new': [model_id]}, self.param_table, refresh=True)
                        else:
                            self.update_resources(model_id)
                time.sleep(interval)
        except Exception as e:
            self.debug.append_stdout("Exception while applying updates from futures: {}\n".format(
                traceback.format_exception(etype=e.__class__, value=e, tb=e.__traceback__)))

    def update_row(self, model_id, data):
        try:
            row_id = self.param_table.get_changed_df().index[model_id]
            for k in data:
                self.param_table._handle_qgrid_msg_helper({
                    'type': 'cell_change',
                    'column': k,
                    'row_index': row_id,
                    'unfiltered_index': model_id,
                    'value': data[k]
                })
            self.param_table._update_table()
        except Exception as e:
            self.debug.append_stdout("Exception while updating a table row : data={} {}\n".format(
                data,
                traceback.format_exception(etype=e.__class__, value=e, tb=e.__traceback__)))            

    def update_resources(self, model_id):
        try:
            self.debug.append_stdout("update_resources - model: {}\n".format(model_id))            
            resource_update = self.get_resource_usage(model_id)
            #self.debug.append_stdout("update_resources - model: {}, data: {}\n".format(model_id, resource_update))
            self.model_resource_plots[model_id].update(resource_update)
        except Exception as e:
            self.debug.append_stdout("Exception while updating model resources plot : model={}, {}\n".format(
                model_id,
                traceback.format_exception(etype=e.__class__, value=e, tb=e.__traceback__)))
    
    def _debug_events(self, event, widget_instance):
        try:
            self.debug.append_stdout("Event received: {}\n".format(event))
        except Exception as e:
            self.debug.append_stdout("Exception while receiving an Event : {}\n".format(
                traceback.format_exception(etype=e.__class__, value=e, tb=e.__traceback__)))

    def display_selected(self, event, widget_instance, refresh=False):
        try:
            self.debug.append_stdout("Event received: {}\n".format(event))

            # this means that all rows have been deselected
            if len(event['new']) == 0:
                return

            row_id = event['new'][0]
            model_id = self.param_table.get_changed_df().index[row_id]
            self._display_plot(model_id, refresh)
            self._display_controls(model_id)
            self._display_resources(model_id)
        except Exception as e:
            self.debug.append_stdout("Exception while updating plot and controls: {}\n".format(
                traceback.format_exception(etype=e.__class__, value=e, tb=e.__traceback__)))

    def _display_plot(self, model_id, refresh=False):
        try:
            self._active_plot = model_id

            # only update the plot if there is more data since the last viewing, or if a new model is running
            if refresh or self.model_data[model_id].num_data_rows > len(self.model_plots[model_id].lines[0].y):
                plot_data = self.model_data[model_id].get_plot_data()
                self.model_plots[model_id].update(plot_data)
            
            with self.plot_output:
                clear_output(wait=True)
                if self.model_displays[model_id] is None:
                    self.model_displays[model_id] = display(self.model_plots[model_id], display_id=True)
                else:
                    update_display(self.model_plots[model_id], display_id=self.model_displays[model_id])
        except Exception as e:
            self.debug.append_stdout("Exception while switching to plot {}: {}\n".format(
                model_id,
                traceback.format_exception(etype=e.__class__, value=e, tb=e.__traceback__))
            )

    def _display_controls(self, model_id):
        try:
            # update buttons and editor
            self.controls.layout.display = 'inherit'
            table_data = self.param_table.get_changed_df()
            row_id = table_data.index[model_id]
            status = table_data.loc[row_id, "status"]
            #self.debug.append_stdout("{}".format(status))
            if status not in ["Stopped", "Ended Training"]:
                self._restart_btn.disabled = True
                self._stop_btn.disabled = False
                self._edit_selected_rows.layout.display = 'none'
            else:
                self._restart_btn.disabled = False
                self._stop_btn.disabled = True
                self._edit_selected_rows.layout.display = 'inherit'            
                param_keys = [k for k in self.compute_params]
                params = table_data.loc[row_id, param_keys].to_dict()
                self._edit_selected_rows.display(row_id, model_id, params)
        except Exception as e:
            self.debug.append_stdout("Exception while updating controls: {}\n".format(
                traceback.format_exception(etype=e.__class__, value=e, tb=e.__traceback__)))

    def _display_resources(self, model_id):
        try:
            with self.resources_output:
                clear_output(wait=True)
                if self.model_resource_displays[model_id] is None:
                    self.model_resource_displays[model_id] = display(self.model_resource_plots[model_id], display_id=True)
                else:
                    update_display(self.model_resource_plots[model_id], display_id=self.model_resource_displays[model_id])
        except Exception as e:
            self.debug.append_stdout("Exception while displaying resources for {}: {}\n".format(
                model_id,
                traceback.format_exception(etype=e.__class__, value=e, tb=e.__traceback__)))
            
    def stop_selected_models(self, event):
        def stop_callback(fut):
            model_id = None
            try:
                model_id = fut.result()

                status = ""
                while status not in ["Stopped", "Ended Training"]:
                    self.update_row(model_id, {"status": "Stopped"})
                    table_data = self.param_table.get_changed_df()
                    row_id = table_data.index[model_id]
                    status = table_data.loc[row_id, "status"]

                self._display_controls(model_id)
            except Exception as e:
                if model_id is not None:
                    self.debug.append_stdout("Exception while stopping Model {} : {}\n".format(
                        model_id,
                        traceback.format_exception(etype=e.__class__, value=e, tb=e.__traceback__)))
                else:
                    self.debug.append_stdout("Exception while stopping Model, future failed: {}\n".format(
                        traceback.format_exception(etype=e.__class__, value=e, tb=e.__traceback__)))
        
        try:
            self.debug.append_stdout("Stop event: {}\n".format(event))
            srows = self.param_table.get_selected_rows()
            self.debug.append_stdout("Stop rows {}\n".format(srows))

            stop_results = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(srows)) as executor:
                for row_id in srows:
                    model_id = self.param_table.get_changed_df().index[row_id]
                    self.update_row(model_id, {"status": "Stopping"})
                    stop_results[model_id] = executor.submit(self._model_controller.stop_model, model_id)

                    # in case the model output updated the table status in between
                    self.update_row(model_id, {"status": "Stopping"})
                    stop_results[model_id].add_done_callback(stop_callback)
                self.debug.append_stdout("Done with stop_selected_models")
        except Exception as e:
            self.debug.append_stdout("Exception while stopping models: {}".format(
                traceback.format_exception(etype=e.__class__, value=e, tb=e.__traceback__)))
    
    def restart_selected_models(self, event):
        try:
            # TODO - check that models are stopped
            self.debug.append_stdout("Restart event: {}\n".format(event))

            srows = self.param_table.get_selected_rows()
            self.debug.append_stdout("Restart rows {}\n".format(srows))
            table_data = self.param_table.get_changed_df()
            param_keys = [k for k in self.compute_params]
            for row_id in srows:
                model_id = table_data.index[row_id]
                self.model_data[model_id] = ModelTaskData(["epoch","loss","val_loss","acc","val_acc"],["status","epoch"])
                edited_params = self._edit_selected_rows.get_values()
                edited_display_values = {
                    "status": "Starting",
                    "epoch": -1,
                    "loss": np.nan,
                    "val_loss": np.nan,
                    "acc": np.nan,
                    "val_acc": np.nan
                }

                for k in edited_params:
                    if isinstance(edited_params[k], list):
                        edited_display_values[k] = str(edited_params[k])
                    else:
                        edited_display_values[k] = edited_params[k]

                self.update_row(model_id, edited_display_values)

                self.debug.append_stdout("Starting new model {} with params {}".format(model_id, edited_params))
                self.debug.append_stdout("Resetting model plot")

                self.model_displays[model_id] = None
                self.model_plots[model_id] = self.vis_func(title="Model {}: {}".format(model_id, edited_params))
                plot_data = self.model_data[model_id].get_plot_data()
                self.model_plots[model_id].update(plot_data)

                self._model_controller.start_model(
                    model_id,
                    self.compute_func,
                    edited_params
                )
                
                if self._active_plot == model_id:
                    self.display_selected({'old': [model_id], 'new': [model_id]}, self.param_table, refresh=True)
                
                self.debug.append_stdout("Updating table status for row {}".format(row_id))
                self.param_table._handle_qgrid_msg_helper({'type': 'selection_changed', 'rows': [row_id]})
        except Exception as e:
            self.debug.append_stdout("Exception while restarting models: {}".format(
                traceback.format_exception(etype=e.__class__, value=e, tb=e.__traceback__)))

    def get_resource_usage(self, model_id):
        return self._model_controller.get_worker_resources(model_id)
    
    def get_models_status(self):
        status = self.param_table.get_changed_df()[["status"]]
        
    def get_model_results(self):
        table_data = self.param_table.get_changed_df()
        params = [k for k in self.compute_params]
        model_results = {}

        for i in range(len(self.model_data)):
            model_results[i] = {
                "parameters": table_data.loc[i, params].to_dict(),
                "history": self.model_data[i].get_plot_data()
            }
        
        return model_results

def _get_ipp_engine_pid():
    import os
    return os.getpid()

class ModelController(object):
    def __init__(self, ipp_cluster_id=None, kale_manager=None):
        if kale_manager is None or len(kale_manager) != 2:
            raise TypeError("Missing Kale Manager details for ModelController, received {}".format(kale_manager))

        self._futures = []
        self._completed = []
        self._active_models = {}
        self._completed_models = {}
        self._ipp_client = ipp.Client(cluster_id=ipp_cluster_id)
        self._ipp_engines = {}
        self._model_id_to_ipp_engine = {}
        
        for x in self._ipp_client.ids:
            self._ipp_engines[x] = {
                "direct_view": self._ipp_client[x],
                "pid": self._ipp_client[x].apply_sync(_get_ipp_engine_pid),
                "busy": False,
                "model_id": -1
            }
        
        self._kale_manager = KaleManagerClient(kale_manager[0], kale_manager[1])
        self._kale_workers = []
        
        for w in self._kale_manager.list_workers():
            try:
                kwc = KaleWorkerClient(w["host"],w["port"])
                self._kale_workers.append(kwc)
            except:
                pass
            
    def _get_available_engine(self):
        for x in self._ipp_engines:
            if self._ipp_engines[x]["busy"]:
                continue
            else:
                return x
        else:
            # all engines are busy, find the least busy one
            min_queue = 2**64
            min_engine = 0
            for x in self._ipp_engines:
                tmp = self._ipp_client.queue_status()[x]['queue']
                if tmp < min_queue:
                    min_queue = tmp
                    min_engine = x
            return min_engine

    def _find_kale_worker_by_model_id(self, model_id):
        if model_id not in self._model_id_to_ipp_engine:
            return None
        
        engine_pid = self._ipp_engines[self._model_id_to_ipp_engine[model_id]]["pid"]
        
        for w in self._kale_workers:
            _tasks = w.get_tasks()
            #print(w.get_tasks()['1']["pid"], engine_pid, w.get_tasks()['1']["pid"] == engine_pid)
            if '1' in _tasks and _tasks['1']["pid"] == engine_pid:
                return w
        else:
            return None

    def start_model(self, model_id, compute_func, params):
        # don't bother if model is running
        if model_id in self._active_models:
            return
        
        #print("Starting model {}".format(model_id))
        x = self._get_available_engine()
        #print("Engine {} is available, submitting".format(x))
        self._futures.append(self._ipp_engines[x]["direct_view"].apply(compute_func, **params))
        self._ipp_engines[x]["model_id"] = model_id
        self._ipp_engines[x]["busy"] = True
        self._active_models[model_id] = len(self._futures) - 1
        self._model_id_to_ipp_engine[model_id] = x

    def stop_model(self, model_id):
        # don't bother if model is not running
        if model_id not in self._active_models:
            return model_id
        
        # get the current number of ipengines
        num_engines = len(self._ipp_engines)
        
        # get the correct worker
        w = self._find_kale_worker_by_model_id(model_id)
        
        #print(self._ipp_client.ids)
        #print(w.get_tasks())
        #print("Stopping engine for model {}".format(model_id))
        
        # stop the ipengine
        w.stop_task('1')
        
        status = w.get_task_status(1)
        while status in ["sleeping", "running"]:
            #print(status)
            time.sleep(0.5)
            status = w.get_task_status('1')
        
        #print(status)
        #print("Engine stopped, cleaning reference and restarting...")
        
        # clean up engine reference
        del self._ipp_engines[self._model_id_to_ipp_engine[model_id]]

        self._ipp_client._unregister_engine({'content': {'id': self._model_id_to_ipp_engine[model_id]}})
        
        # wait for engine to unregister
        engine_ids = self._ipp_client.ids
        while len(engine_ids) == num_engines:
            time.sleep(0.5)
            engine_ids = self._ipp_client.ids
        
        #print(engine_ids)
        
        # start a new ipengine for accepting another model run
        w.start_task('1')
        
        status = w.get_task_status('1')
        while status not in ["sleeping", "running"]:
            #print(status)
            time.sleep(0.5)
            status = w.get_task_status('1')
        
        #print(status)
        #print("Engine restarted")
        
        # refresh ipyparallel engine ids
        engine_ids = self._ipp_client.ids
        #print(engine_ids)
        while len(engine_ids) < num_engines:
            time.sleep(0.5)
            engine_ids = self._ipp_client.ids

        # add engine details
        new_engine_ids = [e for e in self._ipp_client.ids if e not in self._ipp_engines]
        #print(new_engine_ids)
        if len(new_engine_ids) >= 1:
            x = new_engine_ids[0]
        else:
            raise Exception("No new ipengine available!")
        
        del self._model_id_to_ipp_engine[model_id]
        #print("Connecting to ipengine {}".format(x))
        #print("Getting a DirectView to {}".format(x))
        engine_dv = self._ipp_client[x]
        #print("Getting ipengine pid")
        engine_pid = engine_dv.apply_sync(_get_ipp_engine_pid)
        #print("Saving...")
        self._ipp_engines[x] = {
                "direct_view": engine_dv,
                "pid": engine_pid,
                "busy": False,
                "model_id": -1
        }
        #print("Dropping old references, setting stopped model to completed")
        # drop old model reference
        self._futures[self._active_models[model_id]] = None
        del self._active_models[model_id]
        self._completed.append(model_id)
        #print("Done with stop_model")
        return model_id
    
    def set_model_completed(self, model_id):
        if model_id not in self._completed:
            self._completed.append(model_id)
        self._ipp_engines[self._model_id_to_ipp_engine[model_id]]["busy"] = False
    
    def get_completed_models(self):
        return {k: self._futures[self._completed_models[k]] for k in self._completed_models}
    
    def get_running_models(self):
        for i in range(len(self._futures)):
            if self._futures[i] is not None and self._futures[i].done() and i in self._completed:
                self._futures[i] = None
                self._completed_models[i] = self._completed.index(i)
                del self._active_models[i]
        
        return {k: self._futures[self._active_models[k]] for k in self._active_models}
    
    def get_worker_resources(self, model_id):
        w = self._find_kale_worker_by_model_id(model_id)
        if w is not None:
            return w.get_task_resources('1')
        else:
            return {}


class ModelTaskData(object):
    def __init__(self, plot_columns, status_columns):
        super(ModelTaskData, self).__init__()
        
        self._plot_data = ModelPlotTable(plot_columns)
        self._status_data = {k: None for k in status_columns}
        self._updated = True

    @property
    def has_updates(self):
        return self._updated

    @property
    def num_data_rows(self):
        return len(self._plot_data.rows[0])
    
    def get_plot_data(self):
        return self._plot_data.to_dict()
    
    def append_plot_data_row(self, d):
        self._plot_data.append_row(d)
        self._updated = True
    
    def set_status_data(self, d):
        self._status_data.update(d)
        self._updated = True
    
    def get_status_data(self):
        return self._status_data


class ModelPlotTable(object):
    def __init__(self, column_names):
        super(ModelPlotTable, self).__init__()

        self._id = None
        self._num_columns = len(column_names)
        self._num_rows = 0
        self._column_map = {column_names[i]: i for i in range(len(column_names))}
        self._column_data = [list() for c in column_names]
    
    @property
    def columns(self):
        return list(self._column_map.keys())
    
    @property
    def rows(self):
        return self._column_data
        
    def append_column(self, name, vals=None):
        if name in self._column_map:
            raise KeyError("column {} is already in this table".format(name))

        if vals:
            if len(vals) == self._num_rows:
                self._column_data.append(list(vals))
            else:
                raise ValueError("Number of rows must match table")
        else:
            data = [None] * self._num_rows
            self._column_data.append(data)

        self._column_map[name] = len(self._column_data) - 1
        self._updated = True

    def append_row(self, column_data):
        for column_name in self._column_map:
            column_index = self._column_map[column_name]
            if column_name in column_data:
                self._column_data[column_index].append(column_data[column_name])
            else:
                self._column_data[column_index].append(None)

    def to_dict(self):
        return {k: self._column_data[v] for k, v in self._column_map.items()}
