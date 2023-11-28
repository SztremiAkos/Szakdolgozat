import { useEffect, useState } from "react";
import { Navigate, Outlet } from "react-router-dom";
import { auth } from "../constants/firebase";
import { onAuthStateChanged } from "firebase/auth";
import { Box, Toolbar } from "@mui/material";
import Topbar from "../components/common/topbar/Topbar";
import colorConfigs from "../configs/colorConfig";
import sizeConfig from "../configs/sizeConfig";

const ProtectedRoute = () => {
  const [curr, setCurr] = useState(auth.currentUser);
  useEffect(() => {
    const listen = onAuthStateChanged(auth, (user) => {
      if (user) setCurr(user);
      else setCurr(null);
    });
    return () => listen();
  }, []);
  return (
    <Box sx={{ display: "flex" }}>
      <Topbar />
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: `${sizeConfig.sidebar.width}`,
          minHeight: "100vh",
          backgroundColor: colorConfigs.mainBg,
        }}
      >
        <Toolbar />
        {curr ? <Outlet /> : <Navigate to="/login" replace />}
      </Box>
    </Box>
  );
};

export default ProtectedRoute;
