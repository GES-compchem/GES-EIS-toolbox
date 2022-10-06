import random
import numpy as np

from copy import deepcopy
from typing import List, Dict, Tuple

from ges_eis_toolbox.database.data_entry import DataPoint
from ges_eis_toolbox.circuit.circuit_string import CircuitString
from ges_eis_toolbox.database.generator import Generator, Range


class DatasetCreator:
    """
    Dataset creator class capable of building a classification dataset given a set of user
    specified circuits and parameters.

    Parameters
    ----------
    verobse: bool
        if set to True informations about the simulation process will be printed on terminal
    cores: int
        the number of cores to use during the parallel section of the simulation procedure. If
        set to -1 will use all the cores available on the machine.

    """
    def __init__(self, verbose: bool = True, cores: int = -1) -> None:

        self.__verbose = verbose
        self.__cores = cores

        self.__circuits = []
        self.__ranges = []
        self.__X_train, self.__X_test = None, None
        self.__y_train, self.__y_test = None, None

    def add(
        self, circuit_string: CircuitString, ranges: Dict[str, Dict[str, Range]]
    ) -> None:
        """
        Add to the simulation schedule a new circuit generator.

        Parameters
        ----------
        circuit_string: CircuitString
            the CircuitString object encoding the circuit topology
        ranges: Dict[str, Dict[str, Range]]
            the list of ranges associated to each component block type
        """
        
        self.__X_train, self.__X_test = None, None
        self.__y_train, self.__y_test = None, None
        self.__circuits.append(circuit_string)
        self.__ranges.append(ranges)

    def generate(
        self,
        frequency: List[float],
        split_ratio: float = 0.7,
        simulation_limit: int = 100000,
        steps_limit: int = 10,
        equalize: bool = True,
        reshape_to_2D: bool = True,
        polar_form: bool = True
    ) -> None:
        """
        Generates the training and test datasets.

        Parameters
        ----------
        frequency: List[float]
            the list containing the frequency points at which the circuits must be simulated
        split_ratio: float
            the relative fraction of the dataset to be used in the training set
        simulation_limit: int
            sets the maximum number of simulations to be executed by the code (default: 1e+6).
            Please notice how this is different from the real number of simulations that is 
            instead computed based on the circuit structure.
        steps_limit: Union[int, None]
            sets the maximum number of steps for each degree of freedom to be explored in the
            simulation. If set to None, will leave the step number free to approach the value
            that better fits the `simulation_limit` parameter.
        equalize: bool
            if set to True (default), will randomize and cut all the single circuits datasets in order
            to have a balanced dataset in which all the circuit types are equally represented.
        reshape_to_2D: bool
            if set to True (default), will reshape the output dataset to a 2D array having the impedance
            modulus as the first row and the phase as the second one. If set to False the first
            half of the feature vector will be the modulus of the impedance while the second half
            its phase.
        polar_form: bool
            if set to True (default), will represent the impedance spectrum as modulus and phase
            else will adopt the real and imaginary part representation of the spectrum

        """

        # Extract the length of the input dataset
        freq_steps = len(frequency)

        # Run a set of simulations for each circuit type and store the DataPoint objects in memory
        data_points_list: List[List[DataPoint]] = []

        if self.__verbose:
            print("Starting generation of the single-circuit datasets")

        for cs, ranges in zip(self.__circuits, self.__ranges):

            gen = Generator(
                cs, ranges, simulation_limit=simulation_limit, steps_limit=steps_limit
            )

            if self.__verbose:
                print(f"  Running simulations for {str(cs)}:")
                print(f"   -> Number of simulations: {gen.number_of_simulations}")
                print(f"   -> Number of steps: {gen.number_of_steps}")
                print("")

            dp_list = gen.on_the_fly_dataset(frequency, cores=self.__cores)
            data_points_list.append(dp_list)

        # If equalize is true find the minimum length of the single-circuit dataset, randomize the
        # individual ordering and cut each dataset to the same length

        if equalize:

            min_length = min([len(dp_list) for dp_list in data_points_list])

            if self.__verbose:
                print("Running equalization of the single-circuits datasets")
                print(f" -> Minimum number of datapoints: {min_length}")

            buffer = deepcopy(data_points_list)
            data_points_list = []
            for dp in buffer:
                random.shuffle(dp)
                data_points_list.append(dp[0:min_length])

        # Cycle over each set of single-circuit datapoints, extract the spectral data and put
        # them in a list of single-circuit datasets to be used for the machine learning reshaping

        if self.__verbose:
            print("Generating the training examples by category")

        X_list, y_list = [], []
        for i, dp_list in enumerate(data_points_list):

            X = []

            for dp in dp_list:
                
                if polar_form:
                    Z_mod = dp.spectrum.norm_Z
                    Z_phi = dp.spectrum.phi_Z
                    Z = np.concatenate((Z_mod, Z_phi), axis=0)
                else:
                    Z_re = dp.spectrum.real_Z
                    Z_im = dp.spectrum.imag_Z
                    Z = np.concatenate((Z_re, Z_im), axis=0)

                if reshape_to_2D:
                    Z = Z.reshape([2, freq_steps])

                X.append(Z)

            X_list.append(X)
            y_list.append(np.array([i for _ in X]))

        # Concatenate the single-circuit datasets in a single one, randomize the ordering and
        # set the training and validation sets according to the defined split ratio
        X = np.concatenate(X_list, axis=0)
        Y = np.concatenate(y_list, axis=0)
        N_samples = len(Y)

        index_list = [i for i, _ in enumerate(Y)]
        random.shuffle(index_list)

        X_random, Y_random = [], []
        for index in index_list:
            X_random.append(X[index])
            Y_random.append(Y[index])

        if self.__verbose:
            print("Splitting the dataset in training and test sets")

        split = int(split_ratio * N_samples)

        self.__X_train, self.__y_train = np.array(X_random[0:split]), np.array(
            Y_random[0:split]
        )

        if self.__verbose:
            print("  -> Training set shape:")
            print(f"    X: {self.__X_train.shape}")
            print(f"    y: {self.__y_train.shape}")

        self.__X_test, self.__y_test = np.array(X_random[split::]), np.array(
            Y_random[split::]
        )

        if self.__verbose:
            print("  -> Test set shape:")
            print(f"    X: {self.__X_test.shape}")
            print(f"    y: {self.__y_test.shape}")
    
    @property
    def training_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the training set genrated during the call to the generate function.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            the training set in the form of the feature array and the label array
        """
        return self.__X_train, self.__y_train
    
    @property
    def test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the test set genrated during the call to the generate function.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            the test set in the form of the feature array and the label array
        """
        return self.__X_test, self.__y_test
    
    @property
    def number_of_classes(self) -> int:
        """
        Returns the number of classes in the dataset.

        Returns
        -------
        int
            the number of classes in the dataset.
        """
        return len(self.__circuits)
