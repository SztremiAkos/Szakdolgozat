import { BrowserRouter, Routes, Route } from "react-router-dom";
import { getRoutes } from "./routes";
import ProtectedRoute from "./routes/ProtectedRoute";
import LoginPage from "./pages/home/LoginPage";
import Signup from "./pages/home/SignUpPage";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ProtectedRoute/>}>
          {getRoutes}
        </Route>
        <Route path='/login' element={<LoginPage/>}/>
        <Route path='/signup' element={<Signup/>}/>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
