############################################################################################
####                                                                                    ####
####    NAME         : selforganizingmap.py                                             ####
####    EDITED       : Soren Berger             MPI-M       soren.berger@mpimet.mpg.de  ####
####                    - Initial translation from MATLAB to Python                     ####         
####                   Daniel Burt              VLIZ        daniel.burt@vliz.be         ####
####                    - Refactoring of Python code                                    ####
####                    - Adding new functions                                          ####
####    LAST EDIT    : 20.02.2025                                                       ####
####    DESCRIPTION  : This Python implementation is under development within the       ####
####                   Past, Present and Future Marine Climate Change Group of the      ####
####                   Flanders Marine Institute (VLIZ), Belgium.                       ####
####                                                                                    ####
####                   Class file for running Self-Organising Map component of          ####
####                   SOM-FFN method based on the MATLAB implementation of Peter       ####
####                   Landschuetzer and originally described in:                       ####
####                    -  Landschuetzer et al. (2013) Biogeosciences                   ####
####                                                                                    ####
####    DEPENDENCIES : Python 3.12.3                                                    ####
####                    - CartoPy 0.22.0                                                ####
####                    - MatPlotLib 3.6.3                                              ####
####                    - MiniSom 2.3.3 (https://github.com/JustGlowing/minisom)        ####
####                    - NumPy 1.26.4                                                  ####
####                    - SciPy 1.11.4                                                  ####
####                    - Xarray 2024.2.0                                               ####
####                                                                                    ####
####                                                                                    ####
############################################################################################


####  IMPORT PACKAGES
import cartopy as cr
import matplotlib.ticker as tick
import matplotlib.pyplot as plt
import minisom
import numpy as np
import os
import scipy as sp
import time
import warnings
import xarray as xr


####  DEFINE CLASS
class SelfOrganizingMap:

    '''
        NAME        : SelfOrganizingMap
        DESCRIPTION : Python class containing functions related to the
                      Self-Organizing Map component of the SOM-FFN method.
        FUNCTIONS   :   - CalculateMeanMonths            : Calculate the mean of months for 
                                                           time interval to produce mean annual 
                                                           cycle
                        - CountUniqueProvinces           : Counts number of different provinces
                                                           in annual cycle at grid point.
                        - IdentifyProvinces              : Initialises and trains SOM algorithm
                                                           and reshapes winning neurons to
                                                           month, latitude, longitude dimensions
                        - LoadInputData                  : Loads data required for clustering
                        - LoadComparisonProvinces        : Loads stored data files of provinces
                                                           for comparison 
                        - LoadProvinces                  : TODO
                        - ReshapeRearrange               : Reshapes and Rearranges input data
                                                           in preparation for SOM algorithm
                        - PlotProvinces                  : Calls plotting functions prompted by
                                                           function arguments
                        - PlotProvincesMode              : Plot modal and variability province 
                                                           maps
                        - PlotProvincesModeComparison    : Plot modal and variability province 
                                                           maps with comparisons TODO
                        - PlotProvincesMonthly           : Plot monthly province maps
                        - PlotProvincesMonthlyComparison : Plot monthly province maps with
                                                           comparisons TODO
                        - WriteProvinces                 : Write provinces to data file
    '''

    def CalculateMeanMonths(self):
        '''
            NAME        : CalculateMeanMonths
            EDITED      : Daniel Burt       (VLIZ)      20.02.2025
            DESCRIPTION : Calculate the mean of months of input data for entire
                          time period (44 years) to produce a mean annual cycle
                          for every latitude and longitude.
        '''

        # evaluate input array dictionary
        if len(self.input_array_dict) == 0:
            print('ERROR: Input data not found. Please call function: "LoadInputData" before "CalculateMeanMonths".')
            exit()

        # instantiate annual cycle dictionary
        self.annual_cycle_dict = dict()

        # loop through input arrays
        for input_variable in self.input_array_dict.keys():

            # evaluate input variable
            if input_variable != 'lat' and input_variable != 'lon':
        
                # instantiate empty NumPy arrays
                self.annual_cycle_dict[input_variable] = np.empty((12, 180, 360), dtype = 'float32')

                # silence warnings from land points  -->>  FIXME are ALL points land or is it more important to ignore points with NaNs for all months?
                with warnings.catch_warnings():

                    # ignore runtime warnings produced by means of empty slices
                    warnings.simplefilter('ignore', category = RuntimeWarning)

                    # loop through months
                    for month in range(12):

                        # calculate mean of months and disregard NaN values
                        self.annual_cycle_dict[input_variable][month, :, :] = np.nanmean(self.input_array_dict[input_variable][month::12, :, :], axis = 0)


    def CountUniqueProvinces(self):
        '''
            NAME        : CountUniqueProvinces
            EDITED      : Daniel Burt       (VLIZ)      23.01.2025
            DESCRIPTION : Determine the number of occurrences of unique integers
                          in a NumPy array.
        '''

        # initialise output array
        arr_output = np.zeros((180, 360))

        # loop through array
        for i in range(180):
            for j in range(360):

                # determine unique integers in vector
                unique_ints = np.unique(self.provinces[:, i, j])

                # evaluate for NaNs
                if len(unique_ints) == 1 and np.isnan(unique_ints[0]):
                    
                    # propagate NaN values
                    arr_output[i, j] = np.nan
                
                else:

                    # determine number of unique integers found
                    arr_output[i, j] = int(len(unique_ints))

        return arr_output


    def IdentifyProvinces(self, 
                          neuron_map_length = 4, 
                          neuron_map_height = 4, 
                          som_sigma = 2.0, 
                          som_learning_rate = 0.5, 
                          som_neighbourhood_function = 'gaussian',
                          som_topology = 'hexagonal',
                          number_of_epochs = int(1e6)):
        '''
            NAME        : IdentifyProvinces
            EDITED      : Daniel Burt       (VLIZ)      20.01.2025
            DESCRIPTION : Initialise and train the Self Organising Map algorithm
                          using the MiniSom package.
                          Diagnostic time reporting.
            ARGUMENTS   : neuron_map_length               (int)     defines the length of the map neuron layer
                          neuron_map_height               (int)     defines the height of the map neuron layer
                          som_sigma                     (float)     defines the neighbourhood radius
                          som_learning_rate             (float)     ...
                          som_neighbourhood_function    (string)    ...
                          som_topology                  (string)    initial neuron topology of map
                          number_of_epochs              (int)       defines the number of times the neurons are presented with the input vectors
        '''

        # evaluate input array dictionary
        if len(self.input_array_dict) == 0:
            print('ERROR: Input data not found. Please call function: "LoadInputData" before "IdentifyProvinces".')
            exit()

        # evaluate input array dictionary
        if len(self.annual_cycle_dict) == 0:
            print('ERROR: Input data not found. Please call function: "CalculateMeanMonths" before "IdentifyProvinces".')
            exit()

        # initialise time recording for diagnostic reporting
        time_ini = time.time()

        # initialise Self Organising Map algorithm
        self.som_algorithm = minisom.MiniSom(
                                             x = neuron_map_length,
                                             y = neuron_map_height,
                                             input_len = self.arr_som_input.shape[1],
                                             sigma = som_sigma,
                                             learning_rate = som_learning_rate,
                                             neighborhood_function = som_neighbourhood_function,
                                             topology = som_topology,
                                             random_seed = 0
                                             )
        
        # perform first round of training of Self Organising Map Algorithm
        self.som_algorithm.train_random(self.arr_som_input, int(number_of_epochs))

        # end timer and report diagnostic
        time_fin = time.time()
        time_elapsed = time_fin - time_ini
        print(f"Self Organizing Map training ended after {time_elapsed} seconds")

        # allocate winning map neuron two-dimensional location to array as [x, y]
        winning_neurons_position = np.array([self.som_algorithm.winner(x) for x in self.arr_som_input])

        # reformat two-dimensional position tuple to one-dimensional index
        self.arr_winning_neurons = winning_neurons_position[:, 0] * neuron_map_length + winning_neurons_position[:, 1]

        # rearrange winning array neurons into latitude longitude shape for plotting
        self.provinces = np.full((12, 180, 360), np.nan)
        self.provinces = np.ravel(self.provinces)
        self.provinces[~self.arr_nan_index] = self.arr_winning_neurons
        self.provinces = self.provinces.reshape((12, 180, 360))


    def LoadInputData(self, input_dictionary):
        '''
            NAME        : LoadInputData
            EDITED      : Daniel Burt       (VLIZ)      20.02.2025
            DESCRIPTION : Function for loading the data required for clustering.
                          Read MATLAB or netCDF format data files into NumPy arrays
                          in dimensions [months latitude longitude] stored in 
                          dictionary format.
                          Define latitude and longitude arrays for simplicity as 
                          read data files may store in varying formats.
            ARGUMENTS   : input_dictionary      (dict)      input dictionary containing variable name keys and path to data file
        '''

        # evaluate input dictionary
        if len(input_dictionary) == 0:
            print('ERROR: Input data not found. Please give dictionary of variable names and file paths as input to function.')
            exit()

        # instantiate dictionary for input arrays
        self.input_array_dict = dict()

        # loop through input dictionary
        for input_variable in input_dictionary.keys():

            # evaluate given file path
            if not os.path.isfile(input_dictionary[input_variable]):
                print(f"ERROR: File {input_dictionary[input_variable]} for {input_variable} not found. Please enter valid filepath.")
                exit()

            # evaluate data format
            if input_dictionary[input_variable][-4:] == '.mat':
                
                # read matlab data file format
                self.input_array_dict[input_variable] = sp.io.loadmat(input_dictionary[input_variable])[input_variable].astype('float32')

            elif input_dictionary[input_variable][-3:] == '.nc':

                # read netcdf data file format
                self.input_array_dict[input_variable] = xr.load_dataset(input_dictionary[input_variable])[input_variable].values.astype('float32')

        # define latitude and longitude arrays
        self.input_array_dict['lat'] = np.tile(np.linspace( -89.5,  89.5, 180, dtype = np.float32), (360, 1)).T
        self.input_array_dict['lon'] = np.tile(np.linspace(-179.5, 179.5, 360, dtype = np.float32), (180, 1))
    

    def LoadComparisonProvinces(self, fpath_comparison_provinces = None):
        '''
            NAME        : LoadComparisonProvinces
            EDITED      : Daniel Burt   (VLIZ)      16.01.2025
            DESCRIPTION : ...
            ARGUMENTS   : fpath_comparison_provinces    (string)    filepath to valid matlab or netcdf data file of province classes
        '''

        # checK input arguments
        if fpath_comparison_provinces is not None:

            # check existence of file for reading
            if not os.path.isfile(fpath_comparison_provinces):
                print('ERROR: No valid filepath passed to function LoadComparisonProvinces. Please provide valid filepath to load comparison data.')
                exit()
        
        else:
            print('ERROR: No valid filepath passed to function LoadComparisonProvinces. Please provide valid filepath to load comparison data.')
            exit()

        # check filetype from path ending
        if fpath_comparison_provinces[-4:] == '.mat':

            # load matlab data file
            self.comparison_provinces = sp.io.loadmat(fpath_comparison_provinces, appendmat = False)['classes'].squeeze()

        elif fpath_comparison_provinces[-3:] == '.nc':

            # load netCDF data file
            self.comparison_provinces = xr.open_dataset(fpath_comparison_provinces)

        else:
            print('ERROR: Unrecognised filetype for reading comparison data; only MATLAB and netCDF data files are accepted. Please provide valid data filetype.')
            exit()

    
    def PlotInputsMonthly(self):
        '''
            NAME        : PlotInputsMonthly
            EDITED      : Daniel Burt       (VLIZ)      20.02.2025
            DESCRIPTION : Plot monthly maps of input data
        '''

        # loop through input variables
        for input_variable in self.annual_cycle_dict.keys():

            # instantiate projection
            data_crs = cr.crs.PlateCarree()

            # instantiate figure and axes
            fig, axs = plt.subplots(nrows = 4, ncols = 3,
                                    subplot_kw = {'projection': cr.crs.Robinson(central_longitude = 0)},
                                    gridspec_kw = {'wspace': 0.01, 'hspace': -0.08},
                                    figsize = (22, 18.5)
                                    )
            
            # set colour of empty space to white
            fig.patch.set_facecolor('white')

            # flatten axes for simplicity
            axs = np.ravel(axs)

            # loop through monthly data
            for month in range(12):

                # plot contour map of provinces
                plot_province_month = axs[month].pcolormesh(
                                                            self.input_array_dict['lon'],
                                                            self.input_array_dict['lat'],
                                                            self.annual_cycle_dict[input_variable][month, :, :],
                                                            transform = data_crs,
                                                            cmap = plt.cm.get_cmap("jet", 20),
                                                            )
                
            # Add a single colourbar
            cbar = fig.colorbar(plot_province_month, ax = axs, orientation = 'horizontal', fraction = 0.03, pad = 0.01, aspect = 80, shrink = 0.85)

            # modify colourbar labels
            cbar.set_label(input_variable.upper(), fontsize = 20)  # Label for the colorbar
            cbar.ax.tick_params(labelsize = 16)

            # save figure
            plt.savefig(f'./input-data/som-input_plots_{input_variable}', bbox_inches = 'tight', dpi = 100)


    def PlotProvinces(self, plot_type = 'mode-variability', fpath_output_plot = None):
        '''
            NAME        : PlotProvinces
            EDITED      : Daniel Burt       (VLIZ)      03.02.2025
            DESCRIPTION : ...
            ARGUMENTS   : plot_type             (string)    identify which plot function to call
                                                            valid inputs: 'mode-variability'
                                                                          'mode-variability-comparison'
                                                                          'monthly'
                                                                          'monthly-comparison'
                          fpath_output_plot     (string)    filepath for saving output plot
        '''

        # evaluate input array dictionary
        if len(self.input_array_dict) == 0:
            print('ERROR: Input data not found. Please call function: "LoadInputData" before "PlotProvinces".')
            exit()

        # evaluate provinces array
        if not hasattr(self, 'provinces'):
            print('ERROR: Provinces array not found. Please call function: "IdentifyProvinces" before "PlotProvinces".')
            exit()

        # check output plot filepath argument
        if fpath_output_plot is None:
            self.fpath_output_plot = f"./output-plots/som-output_provinces_{plot_type}.png"
        else:
            self.fpath_output_plot = fpath_output_plot

        # check function arguments for function calls
        if plot_type == 'mode-variability':

            self.PlotProvincesMode()

        # if plot_type == 'mode-variability-comparison':

        #     self.PlotProvincesModeComp()

        elif plot_type == 'monthly':

            self.PlotProvincesMonthly()

        # elif plot_type == 'monthly-comparison':

        #     self.PlotProvincesMonthlyComp()

        else:

            print("ERROR: plotting prompt not recognised. Please ented valid prompt from selection: 'mode_variability', 'mode_variability_comparison', 'monthly' or 'monthly_comparison'")
            exit()

    
    def PlotProvincesMode(self):
        '''
            NAME        : PlotProvincesMode
            EDITED      : Daniel Burt       (VLIZ)      20.02.2025
            DESCRIPTION : Plot modal province and province variability maps
        '''

        # calculate modal province
        self.provinces_mode = sp.stats.mode(self.provinces, axis = 0)[0]

        # calculate province variability
        self.provinces_variability = self.CountUniqueProvinces()

        # determine maximum variability
        provinces_variability_vmax = np.nanmax(self.provinces_variability)

        # instantiate projection
        data_crs = cr.crs.PlateCarree()

        # instantiate figure and axes
        fig, axs = plt.subplots(nrows = 2, ncols = 1,
                                subplot_kw = {'projection': cr.crs.Robinson(central_longitude = 0)},
                                gridspec_kw = {'wspace': 0.01, 'hspace': 0.15},
                                figsize = (22, 18.5)
                                )
        
        # set colour of empty space to white
        fig.patch.set_facecolor('white')

        # flatten axes for simplicity
        axs = np.ravel(axs)

        # plot contour map of province mode
        plot_provinces_mode = axs[0].pcolormesh(
                                                self.input_array_dict['lon'],
                                                self.input_array_dict['lat'],
                                                self.provinces_mode[:, :],
                                                transform = data_crs,
                                                cmap = plt.cm.get_cmap("tab20", 16),
                                                vmax = 16,
                                                vmin = 0
                                                )
        
        # configure subplot
        plot_gridlines = axs[0].gridlines(linewidth = 0.5, color = 'k')
        plot_gridlines.xlocator = tick.FixedLocator(range(-180, 181, 30))
        plot_gridlines.ylocator = tick.FixedLocator(range(-90, 91, 15))

        # Add colourbar
        cbar = fig.colorbar(plot_provinces_mode, ax = axs[0], orientation = 'horizontal', fraction = 0.03, pad = 0.02, aspect = 80, shrink = 0.70)

        # modify tick mark position and label
        cbar.set_ticks(np.linspace(0.5, 15.5, 16))
        cbar.set_ticklabels([str(i) for i in range(1, 17)])
        
        # modify colourbar labels
        cbar.set_label('Province Categories', fontsize = 20)  # Label for the colorbar
        cbar.ax.tick_params(labelsize = 16)

        # plot contour map of province mode
        plot_provinces_variability = axs[1].pcolormesh(
                                                       self.input_array_dict['lon'],
                                                       self.input_array_dict['lat'],
                                                       self.provinces_variability[:, :],
                                                       transform = data_crs,
                                                       cmap = plt.cm.get_cmap("tab20", int(provinces_variability_vmax)),
                                                       vmax = provinces_variability_vmax + 1,
                                                       vmin = 1
                                                       )
        
        # configure subplot
        plot_gridlines = axs[1].gridlines(linewidth = 0.5, color = 'k')
        plot_gridlines.xlocator = tick.FixedLocator(range(-180, 181, 30))
        plot_gridlines.ylocator = tick.FixedLocator(range(-90, 91, 15))
            
        # Add colourbar
        cbar = fig.colorbar(plot_provinces_variability, ax = axs[1], orientation = 'horizontal', fraction = 0.03, pad = 0.02, aspect = 80, shrink = 0.70)

        # # modify tick mark position and label
        cbar.set_ticks(np.linspace(1.5, (provinces_variability_vmax + 0.5), int(provinces_variability_vmax)))
        cbar.set_ticklabels([str(i) for i in range(1, int(provinces_variability_vmax + 1))])
        
        # modify colourbar labels
        cbar.set_label('Number of Provinces', fontsize = 20)  # Label for the colorbar
        cbar.ax.tick_params(labelsize = 16)

        plt.savefig(self.fpath_output_plot, bbox_inches = 'tight', dpi = 100)

    
    def PlotProvincesMonthly(self):
        '''
            NAME        : PlotProvincesMonthly
            EDITED      : Daniel Burt       (VLIZ)      20.02.2025
            DESCRIPTION : Plot monthly province maps
        '''

        # instantiate projection
        data_crs = cr.crs.PlateCarree()

        # instantiate figure and axes
        fig, axs = plt.subplots(nrows = 4, ncols = 3,
                                subplot_kw = {'projection': cr.crs.Robinson(central_longitude = 0)},
                                gridspec_kw = {'wspace': 0.01, 'hspace': -0.08},
                                figsize = (22, 18.5)
                                )
        
        # set colour of empty space to white
        fig.patch.set_facecolor('white')

        # flatten axes for simplicity
        axs = np.ravel(axs)

        # loop through monthly data
        for month in range(self.provinces.shape[0]):

            # plot contour map of provinces
            plot_province_month = axs[month].pcolormesh(
                                                        self.input_array_dict['lon'],
                                                        self.input_array_dict['lat'],
                                                        self.provinces[month, :, :],
                                                        transform = data_crs,
                                                        cmap = plt.cm.get_cmap("tab20", 16),
                                                        vmax = 16,
                                                        vmin = 0
                                                        )
            
        # Add a single colourbar
        cbar = fig.colorbar(plot_province_month, ax = axs, orientation = 'horizontal', fraction = 0.03, pad = 0.01, aspect = 80, shrink = 0.85)

        # modify tick mark position and label
        cbar.set_ticks(np.linspace(0.5, 15.5, 16))
        cbar.set_ticklabels([str(i) for i in range(1, 17)])
        
        # modify colourbar labels
        cbar.set_label('Province Categories', fontsize = 20)  # Label for the colorbar
        cbar.ax.tick_params(labelsize = 16)

        plt.savefig(self.fpath_output_plot, bbox_inches = 'tight', dpi = 100)
            

    def ReshapeRearrange(self):
        '''
            NAME        : ReshapeRearrange
            EDITED      : Daniel Burt       (VLIZ)      20.02.2025
            DESCRIPTION : Collapses NumPy arrays into one-dimension, then 
                          identifies and removes NaNs from SOM input array.
        '''

        # evaluate input array dictionary
        if len(self.input_array_dict) == 0:
            print('ERROR: Input data not found. Please call function: "LoadInputData" before "ReshapeRearrange".')
            exit()

        # evaluate input array dictionary
        if len(self.annual_cycle_dict) == 0:
            print('ERROR: Input data not found. Please call function: "CalculateMeanMonths" before "ReshapeRearrange".')
            exit()

        # instantiate list
        annual_cycle_column_list = []
        valid_mask_list          = []

        # loop through annual cycle dictionary
        for variable_name in self.annual_cycle_dict.keys():

            # collapse NumPy arrays of input data into one-dimension for training
            annual_cycle_column_list.append(np.ravel(self.annual_cycle_dict[variable_name]).reshape(-1, 1))

            # store mask of valid data
            valid_mask_list.append(np.isnan(np.ravel(self.annual_cycle_dict[variable_name]).reshape(-1, 1)))


        # create boolean NaN index from all collapsed NumPy arrays with bitwise OR operator
        self.arr_nan_index = np.ravel(np.logical_or.reduce(valid_mask_list))

        # remove NaNs from NumPy arrays and transpose to generate input for SOM algorithm
        arr_som_input_full = np.column_stack(annual_cycle_column_list)
        self.arr_som_input = arr_som_input_full[~self.arr_nan_index]


    def WriteProvinces(self, fpath_output = './input-data', fileext = 'mat'):
        '''
            NAME        : WriteProvinces
            EDITED      : Daniel Burt       (VLIZ)      20.02.2025
            DESCRIPTION : Write identified province map to file.
            ARGUMENT    : fileext       (string)        chosen file extension for output
        '''

        # evaluate input array dictionary
        if len(self.input_array_dict) == 0:
            print('ERROR: Input data not found. Please call function: "LoadInputData" before "WriteProvinces".')
            exit()

        # evaluate provinces array
        if not hasattr(self, 'provinces'):
            print('ERROR: Provinces array not found. Please call function: "IdentifyProvinces" before "WriteProvinces".')
            exit()
        
        # define directory path
        if fpath_output != './input-data':

            # exit in case of invalid directory path
            if not os.path.isdir(fpath_output):
                print('WARNING: Input data directory not found. Please enter valid directory.')
                exit()

            # ensure last character is not /
            if fpath_output[-1] == '/':
                fpath_output = fpath_output[:-1]

        # proceed to writing provinces to data file
        if fileext == 'mat':

            # configure dictionary
            out_dict = {'provinces': self.provinces,
                        'lat': self.input_array_dict['lat'],
                        'lon': self.input_array_dict['lon']
                        }

            # write provinces to MATLAB data file format
            sp.io.savemat(f"{fpath_output}/som-output_provinces.mat", out_dict)
        
        elif fileext == 'nc':

            # configure Xarray DataArray
            DA_out = xr.DataArray(self.provinces,
                                  dims = ['time', 'lat', 'lon'],
                                  coords = {'time': range(self.provinces.shape[0]),
                                            'lat' : np.linspace( -89.5,  89.5, 180, dtype = np.float32), 
                                            'lon' : np.linspace(-179.5, 179.5, 360, dtype = np.float32)},
                                  name = 'provinces')

            # write provinces to netCDF data file format
            DA_out.to_netcdf(f"{fpath_output}/som-output_provinces.nc")

            
        else:

            print('ERROR: Unrecognised file extension. Please enter "mat" or "nc" to write provinces to recognised data file format.')
            exit()
