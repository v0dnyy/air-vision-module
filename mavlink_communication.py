import pymavlink.mavutil as utility
import pymavlink.dialects.v20.all as dialect
from pymavlink import mavutil


class MAVLinkCommunication:

    def __init__(self, port: str = 'COM11', source_system: int = 1, source_component: int = 0):
        self.port = port
        try:
            self.connection = utility.mavlink_connection(
                device=port,
                baud=57600,
                source_system=source_system,
                source_component=source_component,
                autoreconnect=True
            )
            heartbeat = self.connection.wait_heartbeat(timeout=5)
            if not heartbeat:
                raise TimeoutError("Телеметрия не отвечает")
            print(f"MAVLink подключен к {port}")
        except Exception as e:
            print(f"[MAV] Ошибка подключения: {e}")
            self.connection = None

    def send_status(self, text, severity=dialect.MAV_SEVERITY_INFO):
        """
        Универсальная отправка STATUSTEXT.
        """
        if not self.connection:
            print("[MAV] Нет соединения — STATUSTEXT не отправлен")
            return
        text = str(text)[:50]  # Ограничение MAVLink
        msg = dialect.MAVLink_statustext_message(
            severity=severity,
            text=text.encode("utf-8")
        )
        self.connection.mav.send(msg)
        print(f"[MAV] STATUSTEXT: {text}")

    def send_detection_alert(self, detection_count, class_names,
                             severity=dialect.MAV_SEVERITY_ALERT):
        """
        Отправка уведомления о детекции.
        """
        if detection_count > 0:
            msg = f"DETECTED {detection_count}: {', '.join(class_names)}"
        else:
            msg = "Area clear"

        self.send_status(msg, severity)

    def change_to_loiter(self):
        """
        Переключение в LOITER.
        """
        if not self.connection:
            print("[MAV] Нет соединения — режим не изменён")
            return False

        try:
            mode_map = self.connection.mode_mapping()

            if "LOITER" not in mode_map:
                print("[MAV] LOITER недоступен на этом автопилоте")
                return False

            loiter_id = mode_map["LOITER"]

            # Новый корректный способ
            self.connection.mav.set_mode_send(
                self.connection.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                loiter_id
            )

            print("[MAV] Режим изменён → LOITER")
            return True

        except Exception as e:
            print(f"[MAV] Ошибка при смене режима: {e}")
            return False

    def close(self):
        if self.connection:
            self.connection.close()
            print("MAVLink соединение закрыто")