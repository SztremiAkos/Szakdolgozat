import { AppBar, Box, Button, Toolbar, Typography } from "@mui/material";
import sizeConfig from "../../../configs/sizeConfig";
import colorConfigs from "../../../configs/colorConfig";
import { useEffect, useState } from "react";
import { User, onAuthStateChanged, signOut } from "firebase/auth";
import { auth } from "../../../constants/firebase";
import { useNavigate } from "react-router-dom";

const Topbar = () => {
  const colorConfig = colorConfigs.topbar;
  const navigate = useNavigate();
  const title = "Rendszeridentifikáció mélytanulásos hálók segítségével";
  const [user, setUser] = useState<User | null>();
  useEffect(() => {
    const listen = onAuthStateChanged(auth, (user) => {
      if (user) setUser(user);
      else setUser(null);
    });
    return () => listen();
  }, []);
  const logOut = async () => {
    await signOut(auth);
    navigate("/");
  };
  return (
    <AppBar
      position="fixed"
      sx={{
        width: `100%`,
        ml: sizeConfig.sidebar.width,
        boxShadow: "unset",
        backgroundColor: colorConfig.bg,
        color: colorConfig.color,
      }}
    >
      <Toolbar sx={{ justifyContent: "space-between" }}>
        <Typography variant="h6">{title}</Typography>
        {user && (
          <Box sx={{display: 'flex', gap:2}}>
            <Typography variant="h6">Logged in as {user.email}</Typography>
            <Button variant='outlined' onClick={() => logOut()}>Log out</Button>
          </Box>
        )}
      </Toolbar>
    </AppBar>
  );
};

export default Topbar;
