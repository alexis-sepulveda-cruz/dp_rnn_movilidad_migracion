from abc import ABC, abstractmethod
import typing as t


class ModelBuilderPort(ABC):
    """Puerto para construir modelos de predicción."""

    @abstractmethod
    def build_model(self, input_shape: tuple) -> t.Any:
        """
        Construye un modelo con la forma de entrada especificada.

        Args:
            input_shape: Forma de los datos de entrada

        Returns:
            Un modelo construido
        """
        pass