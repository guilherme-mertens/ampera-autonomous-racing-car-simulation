import numpy as np


class Filters:
    """
    Classe para aplicar filtros em cones detectados.
    """

    def __init__(self, steer_lim=0.2):
        """
        Inicializa a classe Filtros.
        
        Args:
            steer_lim (float): Limite de direção para aplicar filtros específicos.
        """
        self.steer_lim = steer_lim

    def remove_far_cones(self, colored_cones, steering, x_delimit=3.5, x_slam_delimit=4.8, y_slam_delimit=9.5):
        """
        Remove cones distantes com base na direção do veículo.

        Args:
            colored_cones (np.ndarray): Array de cones detectados.
            steering (float): Valor da direção atual do veículo.
            x_delimit (float): Limite para filtragem no eixo x.
            x_slam_delimit (float): Limite SLAM para filtragem no eixo x.
            y_slam_delimit (float): Limite SLAM para filtragem no eixo y.

        Returns:
            np.ndarray: Cones filtrados.
        """
        if steering > 0.0001:
            distance_x_idx = np.where(colored_cones[:, 0] > -x_delimit)[0]
            filtered_cones = colored_cones[distance_x_idx]

            x_slam_idx = np.where(filtered_cones[:, 0] < x_slam_delimit)[0]
            filtered_cones = filtered_cones[x_slam_idx]

            y_slam_idx = np.where(filtered_cones[:, 1] < y_slam_delimit)[0]
            filtered_cones = filtered_cones[y_slam_idx]
        elif steering < -0.0001:
            distance_x_idx = np.where(colored_cones[:, 0] < x_delimit)[0]
            filtered_cones = colored_cones[distance_x_idx]

            x_slam_idx = np.where(filtered_cones[:, 0] > -x_slam_delimit)[0]
            filtered_cones = filtered_cones[x_slam_idx]

            y_slam_idx = np.where(filtered_cones[:, 1] < y_slam_delimit)[0]
            filtered_cones = filtered_cones[y_slam_idx]
        else:
            filtered_cones = colored_cones

        return filtered_cones

    def find_main_road_blue(self, blue_cones_detected, yellow_cones_detected):
        """
        Encontra a estrada principal a partir dos cones azuis.

        Args:
            blue_cones_detected (np.ndarray): Array de cones azuis detectados.
            yellow_cones_detected (np.ndarray): Array de cones amarelos detectados.

        Returns:
            tuple: Índices ordenados dos cones amarelos e os cones azuis filtrados.
        """
        yellow_cones_detected = yellow_cones_detected[:, 0:2]
        dist_yellow = np.sqrt(yellow_cones_detected[:, 0] ** 2 + yellow_cones_detected[:, 1] ** 2).argsort()

        x1, y1 = yellow_cones_detected[dist_yellow[0]]
        yellow_cones_detected = np.delete(yellow_cones_detected, dist_yellow[0], 0)

        x2, y2 = yellow_cones_detected[yellow_cones_detected[:, 0].argsort()][0]

        a = np.array([[1, x1], [1, x2]])
        b = np.array([y1, y2])
        x = np.linalg.solve(a, b)

        f_x_blue = blue_cones_detected[:, 0] * x[1] + x[0]
        y_blue = blue_cones_detected[:, 1]

        excluir_idx = np.where(y_blue > f_x_blue)[0]
        blue_cones = np.delete(blue_cones_detected, excluir_idx, 0)
        
        return dist_yellow, blue_cones

    def find_main_road_yellow(self, blue_cones_detected, yellow_cones_detected):
        """
        Encontra a estrada principal a partir dos cones amarelos.

        Args:
            blue_cones_detected (np.ndarray): Array de cones azuis detectados.
            yellow_cones_detected (np.ndarray): Array de cones amarelos detectados.

        Returns:
            tuple: Índices ordenados dos cones azuis e os cones amarelos filtrados.
        """
        blue_cones_detected = blue_cones_detected[:, 0:2]

        dist_blue = np.sqrt(blue_cones_detected[:, 0] ** 2 + blue_cones_detected[:, 1] ** 2).argsort()

        x1, y1 = blue_cones_detected[dist_blue[0]]
        blue_cones_detected = np.delete(blue_cones_detected, dist_blue[0], 0)

        x2, y2 = blue_cones_detected[blue_cones_detected[:, 0].argsort()][-1]

        a = np.array([[1, x1], [1, x2]])
        b = np.array([y1, y2])
        x = np.linalg.solve(a, b)

        f_x_yellow = yellow_cones_detected[:, 0] * x[1] + x[0]
        y_yellow = yellow_cones_detected[:, 1]

        excluir_idx = np.where(y_yellow > f_x_yellow)[0]
        yellow_cones = np.delete(yellow_cones_detected, excluir_idx, 0)
        
        return dist_blue, yellow_cones

    def find_current_lane(self, colored_cones, steering, max_cones=4):
        """
        Encontra a pista atual baseada na direção e no número de cones detectados.

        Args:
            colored_cones (np.ndarray): Array de cones detectados.
            steering (float): Valor da direção atual do veículo.
            max_cones (int): Número máximo de cones para considerar a filtragem.

        Returns:
            np.ndarray: Cones filtrados.
        """
        yellow_cones_idx = np.where(colored_cones[:, -1] == 2)[0]
        blue_cones_idx = np.where(colored_cones[:, -1] == 0)[0]

        yellow_cones_detected = colored_cones[yellow_cones_idx]
        blue_cones_detected = colored_cones[blue_cones_idx]

        if len(yellow_cones_idx) != 0 and len(blue_cones_idx) != 0:
            if len(blue_cones_detected) > max_cones or len(yellow_cones_detected) > max_cones:
                if steering < -self.steer_lim:
                    dist_yellow, blue_cones_filtered = self.find_main_road_blue(blue_cones_detected, yellow_cones_detected)
                    if len(blue_cones_filtered) == 0:
                        blue_improv = colored_cones[yellow_cones_idx][dist_yellow[:2]]
                        blue_improv[:, 0] -= 4
                        filtered_cones = np.delete(colored_cones, blue_cones_idx, 0)
                        filtered_cones = np.append(filtered_cones, blue_improv, axis=0)
                    else:
                        filtered_cones = np.delete(colored_cones, blue_cones_idx, 0)
                        filtered_cones = np.append(filtered_cones, blue_cones_filtered, axis=0)
                elif steering > self.steer_lim:
                    dist_blue, yellow_cones_filtered = self.find_main_road_yellow(blue_cones_detected, yellow_cones_detected)
                    if len(yellow_cones_filtered) == 0:
                        yellow_improv = colored_cones[blue_cones_idx][dist_blue[:2]]
                        yellow_improv[:, 0] += 4
                        filtered_cones = np.delete(colored_cones, yellow_cones_idx, 0)
                        filtered_cones = np.append(filtered_cones, yellow_improv, axis=0)
                    else:
                        filtered_cones = np.delete(colored_cones, yellow_cones_idx, 0)
                        filtered_cones = np.append(filtered_cones, yellow_cones_filtered, axis=0)
                else:
                    filtered_cones = colored_cones
            else:
                filtered_cones = colored_cones
        else:
            filtered_cones = colored_cones

        return filtered_cones

    def no_blue(self, colored_cones):
        """
        Adiciona cones virtuais azuis se nenhum for detectado.

        Args:
            colored_cones (np.ndarray): Array de cones detectados.

        Returns:
            np.ndarray: Cones com cones virtuais adicionados se necessário.
        """
        blue_cones_idx = np.where(colored_cones[:, -1] == 0)[0]

        if len(blue_cones_idx) == 0:
            yellow_cones_idx = np.where(colored_cones[:, -1] == 2)[0]
            yellow_cones_detected = colored_cones[yellow_cones_idx]
            yellow_cones_detected_0 = yellow_cones_detected[:, :2]

            dist_yellow = np.sqrt(yellow_cones_detected_0[:, 0] ** 2 + yellow_cones_detected_0[:, 1] ** 2).argsort()
            blue_improv = yellow_cones_detected[dist_yellow[:2]]
            blue_improv[:, 0] -= 4
            blue_improv[:, -1] = 0
            colored_cones = np.append(colored_cones, blue_improv, axis=0)

        return colored_cones

    def no_yellow(self, colored_cones):
        """
        Adiciona cones virtuais amarelos se nenhum for detectado.

        Args:
            colored_cones (np.ndarray): Array de cones detectados.

        Returns:
            np.ndarray: Cones com cones virtuais adicionados se necessário.
        """
        yellow_cones_idx = np.where(colored_cones[:, -1] == 2)[0]

        if len(yellow_cones_idx) == 0:
            blue_cones_idx = np.where(colored_cones[:, -1] == 0)[0]
            blue_cones_detected = colored_cones[blue_cones_idx]
            blue_cones_detected_0 = blue_cones_detected[:, :2]

            dist_blue = np.sqrt(blue_cones_detected_0[:, 0] ** 2 + blue_cones_detected_0[:, 1] ** 2).argsort()
            yellow_improv = blue_cones_detected[dist_blue[:2]]
            yellow_improv[:, 0] += 4
            yellow_improv[:, -1] = 2
            colored_cones = np.append(colored_cones, yellow_improv, axis=0)

        return colored_cones
