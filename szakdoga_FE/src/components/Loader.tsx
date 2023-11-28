import CircularProgress from "@mui/material/CircularProgress";
import Box from "@mui/material/Box";

const Loader = () => {
  return (
    <Box
      display="flex"
      alignItems="center"
      justifyContent="center"
      position="fixed"
      top={0}
      left={0}
      width="100%"
      height="100%"
      zIndex="modal"
      bgcolor="rgba(255, 255, 255,0.4)"
    >
      <CircularProgress />
    </Box>
  );
};

export default Loader;
