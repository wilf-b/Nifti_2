"""
RunTag carries the metadata for building output filenames. It replaces
the six mutable module globals baseline_code.py used, so nothing in the
simulation logic touches global state to name a file.
"""

from dataclasses import dataclass


@dataclass
class RunTag:
    # added by Wilf
    """
    Filename metadata. mode is the run kind ("[Static]", "[Controlled]",
    ...), track_currents is a description like "Row 1: Default", pod_angle
    a string like "[Theta 0_0, 0_0, 0_0]".
    """

    mode: str = ""
    track_name: str = ""
    track_currents: str = ""
    pod_name: str = ""
    pod_angle: str = ""

    @classmethod
    def from_euler(cls, mode, track_name, track_currents, pod_name,
                   roll, pitch, yaw):
        # added by Wilf
        """Build a RunTag, formatting the Euler angles into the filename style."""
        angle_str = f"[Theta {roll}, {pitch}, {yaw}]".replace(".", "_")
        return cls(mode=mode, track_name=track_name,
                   track_currents=track_currents, pod_name=pod_name,
                   pod_angle=angle_str)

    def full_tag(self):
        # added by Wilf
        """mode, track, currents, pod, angle. Used by most saves."""
        return (
            f"{self.mode} Track ({self.track_name}) "
            f"Track Current ({self.track_currents}) "
            f"Pod ({self.pod_name} {self.pod_angle})"
        )

    def comparison_tag(self):
        # added by Wilf
        """Tag for ideal-vs-controlled comparison files."""
        return (
            f"[Comparison] Track ({self.track_name}) "
            f"Track Current ({self.track_currents}) "
            f"Pod ({self.pod_name} {self.pod_angle})"
        )
