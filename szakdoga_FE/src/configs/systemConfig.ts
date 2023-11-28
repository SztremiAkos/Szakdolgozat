type System = {
  name: string;
  paramLabels: string[];
};

const systems: System[] = [
  {
    name: "Mass Spring Damper",
    paramLabels: ["x", "y", "z"],
  },
  {
    name: "Inverted Pendulum",
    paramLabels: ["y", "z"],
  },
];

export default systems;
